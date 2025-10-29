import copy
import os
import re
import threading
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models

from clip.clip.model import CLIP, AttentionPool2d, ModifiedResNet, build_model
from resnet_features import conv1x1

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

    def _save_outputs_hook(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def conv_info(self):
        raise NotImplementedError


class ResNetFeature(GeneralFeature):
    def __init__(self, model: models.ResNet):
        super().__init__(model, model.avgpool)

    def _save_outputs_hook(self):
        def fn(module: nn.Module, input: Tuple[torch.Tensor]):
            with self.lock:
                self._features[input[0].device] = input[0]
        return fn

    def forward(self, x: torch.Tensor):
        _ = self.model(x)
        with self.lock:
            x = self._features[x.device]
        return x

    def conv_info(self):
        kernel_sizes = [7, 3]
        strides = [2, 2]
        paddings = [3, 1]
        convs = [layer for name, layer in list(
            self.model.named_modules()) if re.search(r'layer.*?conv', name)]
        kernel_sizes.extend([layer.kernel_size for layer in convs])
        strides.extend([layer.stride for layer in convs])
        paddings.extend([layer.padding for layer in convs])
        kernel_sizes = [x[0] if isinstance(
            x, tuple) else x for x in kernel_sizes]
        strides = [x[0] if isinstance(x, tuple) else x for x in strides]
        paddings = [x[0] if isinstance(x, tuple) else x for x in paddings]
        return kernel_sizes, strides, paddings


class CLIPResNetFeature(GeneralFeature):
    def __init__(self, model: CLIP):
        assert type(model.visual) == ModifiedResNet
        assert type(model.visual.attnpool) == AttentionPool2d
        image_model = copy.deepcopy(model.visual)
        image_model.attnpool = nn.Identity()
        super().__init__(image_model, None)
        self.attnpool = copy.deepcopy(model.visual.attnpool)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x

    def adapt(self, x: torch.Tensor):
        x_size = x.size()  # [BS, 2048, 7, 7]
        global_x = self.attnpool(x)  # [BS, 1024]
        x = x.permute(0, 2, 3, 1)  # [BS, 7, 7, 2048]
        x = x.reshape(-1, x_size[1])  # [BS x 7 x 7, 2048]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [BS x 7 x 7, 2048, 1, 1]
        x = x.expand(-1, -1, 7, 7)  # [BS x 7 x 7, 2048, 7, 7]
        x = self.attnpool(x)  # [BS x 7 x 7, 1024]
        x = x.reshape(x_size[0], x_size[2], x_size[3], -1)  # [BS, 7, 7, 1024]
        local_x = x.permute(0, 3, 1, 2)  # [BS, 1024, 7, 7]

        return global_x, local_x

    def conv_info(self):
        kernel_sizes = [7, 3]
        strides = [2, 2]
        paddings = [3, 1]
        convs = [layer for name, layer in list(
            self.model.named_modules()) if re.search(r'layer.*?conv', name)]
        kernel_sizes.extend([layer.kernel_size for layer in convs])
        strides.extend([layer.stride for layer in convs])
        paddings.extend([layer.padding for layer in convs])
        kernel_sizes = [x[0] if isinstance(
            x, tuple) else x for x in kernel_sizes]
        strides = [x[0] if isinstance(x, tuple) else x for x in strides]
        paddings = [x[0] if isinstance(x, tuple) else x for x in paddings]
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


def resnet34_features(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    torch.hub.set_dir(model_dir)
    model = models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None, **kwargs)
    model = ResNetFeature(model)
    return model


def resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    torch.hub.set_dir(model_dir)
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None, **kwargs)
    model = ResNetFeature(model)
    return model


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


if __name__ == '__main__':

    r34_features = resnet34_features(pretrained=True)
    print(r34_features)

    r50_features = resnet50_features(pretrained=True)
    print(r50_features)

    clip_r50_features = clip_resnet50_features(pretrained=True)
    print(clip_r50_features)
