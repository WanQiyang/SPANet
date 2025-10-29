import copy
import functools
import os
import re
import threading
from typing import Dict, Union

import torch
import torch.nn as nn

from clip.clip.model import (CLIP, AttentionPool2d, ModifiedResNet,
                             VisionTransformer, build_model)

model_dir = './pretrained_models'


def convert_weights_to_fp32(model: nn.Module):
    """Convert applicable model parameters to fp32"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)


class GeneralFeature(nn.Module):
    def __init__(self, model: nn.Module, hook_layer: Union[nn.Module, str, None]):
        super().__init__()
        self.model = model
        self._features: Dict[str, torch.Tensor] = dict()
        if hook_layer is not None:
            self.hook_layer = hook_layer if isinstance(hook_layer, nn.Module) else dict([
                *model.named_modules()])[hook_layer]
            self.hook = self.hook_layer.register_forward_pre_hook(
                self._save_outputs_hook())
            self.lock = threading.Lock()
            # TODO: do something to remove hook on delete

    def _save_outputs_hook(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def conv_info(self):
        raise NotImplementedError


class CLIPResNetFeature(GeneralFeature):
    def __init__(self, model: CLIP):
        assert type(model.visual) == ModifiedResNet
        assert type(model.visual.attnpool) == AttentionPool2d
        image_model = copy.deepcopy(model.visual)
        image_model.attnpool = nn.Identity()
        super().__init__(image_model, None)
        self.attnpool = copy.deepcopy(model.visual.attnpool)
        # TODO: adapt ViT backbone

    def forward(self, x: torch.Tensor):
        f = self.model(x)
        f_size = f.size()  # [BS, 2048, 7, 7]
        g = self.attnpool(f)  # [BS, 1024]
        x = f.permute(0, 2, 3, 1)  # [BS, 7, 7, 2048]
        x = x.reshape(-1, f_size[1])  # [BS x 7 x 7, 2048]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [BS x 7 x 7, 2048, 1, 1]
        x = x.expand(-1, -1, 7, 7)  # [BS x 7 x 7, 2048, 7, 7]
        x = self.attnpool(x)  # [BS x 7 x 7, 1024]
        x = x.reshape(f_size[0], f_size[2], f_size[3], -1)  # [BS, 7, 7, 1024]
        x = x.permute(0, 3, 1, 2)  # [BS, 1024, 7, 7]
        return f, g, x

    def conv_info(self):
        kernel_sizes = [7, 3]
        strides = [2, 2]
        paddings = [3, 1]
        # convs = [layer for name, layer in list(
        #     self.model.named_modules()) if re.search(r'layer.*?conv', name)]
        # kernel_sizes.extend([layer.kernel_size for layer in convs])
        # strides.extend([layer.stride for layer in convs])
        # paddings.extend([layer.padding for layer in convs])

        # kernel_sizes = [x[0] if isinstance(
        #     x, tuple) else x for x in kernel_sizes]
        # strides = [x[0] if isinstance(x, tuple) else x for x in strides]
        # paddings = [x[0] if isinstance(x, tuple) else x for x in paddings]

        layers = [layer for name, layer in list(
            self.model.named_modules()) if re.search(r'layer.*?(conv|avgpool)', name)]

        for layer in layers:
            if type(layer) == nn.Conv2d:
                kernel_sizes.append(layer.kernel_size[0])
                strides.append(layer.stride[0])
                paddings.append(layer.padding[0])
            elif type(layer) == nn.AvgPool2d:
                kernel_sizes.append(1)
                strides.append(layer.stride)
                paddings.append(0)
            elif type(layer) == nn.Identity:
                kernel_sizes.append(1)
                strides.append(1)
                paddings.append(0)
            else:
                raise

        return kernel_sizes, strides, paddings


def _vit_modified_forward(self: VisionTransformer, x: torch.Tensor):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    # shape = [*, width, grid ** 2]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                  dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # x = self.ln_post(x[:, 0, :])

    # if self.proj is not None:
    #     x = x @ self.proj

    return x


class CLIPViTFeature(GeneralFeature):
    def __init__(self, model: CLIP, pool2d: int = None):
        assert type(model.visual) == VisionTransformer
        image_model = copy.deepcopy(model.visual)
        self.feature_map_size = int(
            (image_model.positional_embedding.size(0) - 1) ** 0.5)  # 7 or 14
        image_model.forward = functools.partial(
            _vit_modified_forward, image_model)
        super().__init__(image_model, None)
        self.pool2d = nn.AvgPool2d(kernel_size=(
            pool2d, pool2d), stride=pool2d) if pool2d else None

    def forward(self, x: torch.Tensor):
        x_size = x.size()  # [BS, 3, 224, 224]
        x = self.model(x)  # x [BS, 50, 768] or [BS, 1+14*14, 768]

        # TODO: ln_post should be here?
        x = self.model.ln_post(x)  # x [BS, 50, 768] or [BS, 1+14*14, 768]

        # TODO: use unproj f? or f = the final x
        f = x[:, 1:, :]  # f [BS, 49, 768] or [BS, 14*14, 768]
        f = f.permute(0, 2, 1)  # f[BS, 768, 49] or [BS, 768, 14*14]
        f = f.reshape(x_size[0], -1, self.feature_map_size,
                      self.feature_map_size)  # f [BS, 768, 7, 7] or [BS, 768, 14, 14]
        if self.pool2d:
            f = self.pool2d(f)  # f [BS, 768, 7, 7]

        if self.model.proj is not None:
            x = x @ self.model.proj  # x [BS, 50, 512] or [BS, 1+14*14, 512]

        g = x[:, 0, :]  # g [BS, 512]
        x = x[:, 1:, :]  # f [BS, 49, 512] or [BS, 14*14, 512]
        x = x.permute(0, 2, 1)  # [BS, 512, 49] or [BS, 512, 14*14]
        x = x.reshape(x_size[0], -1, self.feature_map_size,
                      self.feature_map_size)  # x [BS, 512, 7, 7] or [BS, 512, 14, 14]

        if self.pool2d:
            x = self.pool2d(x)  # x[BS, 512, 7, 7]

        return f, g, x

    def conv_info(self):
        # TODO
        kernel_sizes = [1,]
        strides = [1,]
        paddings = [0,]
        return kernel_sizes, strides, paddings


class CLIPTextFeature(nn.Module):
    def __init__(self, model: CLIP):
        super().__init__()
        self.dtype = model.dtype
        self.token_embedding = copy.deepcopy(model.token_embedding)
        self.positional_embedding = copy.deepcopy(model.positional_embedding)
        self.transformer = copy.deepcopy(model.transformer)
        self.ln_final = copy.deepcopy(model.ln_final)
        self.text_projection = copy.deepcopy(model.text_projection)

    # same as CLIP encode_text
    def forward(self, text):
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)
              ] @ self.text_projection

        return x


def clip_resnet50_features(pretrained=False, **kwargs):
    if pretrained:
        pretrained_file = os.path.join(model_dir, 'clip', 'RN50.pt')
        assert os.path.exists(
            pretrained_file), 'Please download the CLIP model first'
        with open(pretrained_file, "rb") as opened_file:
            try:
                # loading JIT archive
                clip = torch.jit.load(opened_file, map_location="cpu").eval()
                state_dict = None
            except RuntimeError:
                # loading saved state dict
                state_dict = torch.load(opened_file, map_location="cpu")

        state_dict = state_dict or clip.state_dict()

        clip = build_model(state_dict)
        convert_weights_to_fp32(clip)
        # clip.train()
    else:
        raise NotImplementedError

    image_model = CLIPResNetFeature(clip)
    text_model = CLIPTextFeature(clip)
    del clip

    return image_model, text_model


def clip_resnet101_features(pretrained=False, **kwargs):
    if pretrained:
        pretrained_file = os.path.join(model_dir, 'clip', 'RN101.zip')
        assert os.path.exists(
            pretrained_file), 'Please download the CLIP model first'
        with open(pretrained_file, "rb") as opened_file:
            try:
                # loading JIT archive
                clip = torch.jit.load(opened_file, map_location="cpu").eval()
                state_dict = None
            except RuntimeError:
                # loading saved state dict
                state_dict = torch.load(opened_file, map_location="cpu")

        state_dict = state_dict or clip.state_dict()

        clip = build_model(state_dict)
        convert_weights_to_fp32(clip)
        # clip.train()
    else:
        raise NotImplementedError

    image_model = CLIPResNetFeature(clip)
    text_model = CLIPTextFeature(clip)
    del clip

    return image_model, text_model


def clip_vitb32_features(pretrained=False, **kwargs):
    if pretrained:
        pretrained_file = os.path.join(model_dir, 'clip', 'ViT-B-32.zip')
        assert os.path.exists(
            pretrained_file), 'Please download the CLIP model first'
        with open(pretrained_file, "rb") as opened_file:
            try:
                # loading JIT archive
                clip = torch.jit.load(opened_file, map_location="cpu").eval()
                state_dict = None
            except RuntimeError:
                # loading saved state dict
                state_dict = torch.load(opened_file, map_location="cpu")

        state_dict = state_dict or clip.state_dict()

        clip = build_model(state_dict)
        convert_weights_to_fp32(clip)
        # clip.train()
    else:
        raise NotImplementedError

    image_model = CLIPViTFeature(clip)
    text_model = CLIPTextFeature(clip)
    del clip

    return image_model, text_model


def clip_vitb16_features(pretrained=False, **kwargs):
    if pretrained:
        pretrained_file = os.path.join(model_dir, 'clip', 'ViT-B-16.zip')
        assert os.path.exists(
            pretrained_file), 'Please download the CLIP model first'
        with open(pretrained_file, "rb") as opened_file:
            try:
                # loading JIT archive
                clip = torch.jit.load(opened_file, map_location="cpu").eval()
                state_dict = None
            except RuntimeError:
                # loading saved state dict
                state_dict = torch.load(opened_file, map_location="cpu")

        state_dict = state_dict or clip.state_dict()

        clip = build_model(state_dict)
        convert_weights_to_fp32(clip)
        # clip.train()
    else:
        raise NotImplementedError

    image_model = CLIPViTFeature(clip, pool2d=2)
    text_model = CLIPTextFeature(clip)
    del clip

    return image_model, text_model


if __name__ == '__main__':

    clip_r50_features = clip_resnet50_features(pretrained=True)
    print(clip_r50_features)

    clip_r101_features = clip_resnet101_features(pretrained=True)
    print(clip_r101_features)

    clip_vb32_features = clip_vitb32_features(pretrained=True)
    print(clip_vb32_features)

    clip_vb16_features = clip_vitb16_features(pretrained=True)
    print(clip_vb16_features)
