# SPANet test code

import os
from collections import OrderedDict

import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import model

dataset_path = './datasets/CUB'
test_settings = {
    'ResNet-50': {
        'base_architecture': 'clip_resnet50',
        'model_path': './my_models/CUB_RN50.pth',
        'prototype_shape': (2000, 2048, 1, 1),
    },
    'ResNet-101': {
        'base_architecture': 'clip_resnet101',
        'model_path': './my_models/CUB_RN101.pth',
        'prototype_shape': (2000, 2048, 1, 1),
    },
    'ViT-B/32': {
        'base_architecture': 'clip_vitb32',
        'model_path': './my_models/CUB_ViTB32.pth',
        'prototype_shape': (2000, 768, 1, 1),
    },
    'ViT-B/16': {
        'base_architecture': 'clip_vitb16',
        'model_path': './my_models/CUB_ViTB16.pth',
        'prototype_shape': (2000, 768, 1, 1),
    },
}

for model_name, setting in test_settings.items():
    base_architecture = setting['base_architecture']
    model_path = setting['model_path']
    prototype_shape = setting['prototype_shape']

    # inference settings
    device = 'cuda:0'  # cpu or cuda
    batch_size = 160

    # model settings
    img_size = 224
    num_classes = 200
    prototype_activation_function = 'log'
    add_on_layers_type = 'regular'

    # load model
    state_dict = torch.load(model_path, map_location=device)
    sprnet = model.construct_SPRNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)

    processed_state_dict = OrderedDict()
    for k, v in state_dict.items():
        assert k.startswith('module.')
        processed_state_dict[k[7:]] = v

    sprnet.load_state_dict(processed_state_dict)
    sprnet = sprnet.to(device)
    sprnet.eval()

    # construct dataset
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    test_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_dataset = datasets.ImageFolder(root=os.path.join(
        dataset_path, 'cub200_cropped', 'test_cropped'), transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    # run test
    n_examples = 0
    n_correct = 0

    for input, target in tqdm(test_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            output, _, _, _ = sprnet.image_embed(input)
            _, predicted = torch.max(output.data, 1)

        n_examples += target.size(0)
        n_correct += (predicted == target).sum()

    print('model name: {}, test accuracy: {:.2f}.'.format(
        model_name, n_correct / n_examples * 100))

print('test completed. have a nice day!')
