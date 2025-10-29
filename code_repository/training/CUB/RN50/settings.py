exp_name = 'cub-rn50-0000'

base_architecture = 'clip_resnet50'
img_size = 224
prototype_shape = (2000, 2048, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

dataset_path = './datasets/CUB'
train_batch_size = 80
test_batch_size = 160

joint_optimizer_lrs = {'features': 1e-5,
                       'add_on_layers': 3e-4,
                       'prototype_vectors': 3e-4,
                       'clip': 1e-5}
joint_lr_step_size = 10
joint_lr_step_gamma = 0.1

warm_optimizer_lrs = {'add_on_layers': 3e-4,
                      'prototype_vectors': 3e-4}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'global_clip': 0.1,
    'local_clip': 0.1,
}

num_train_epochs = 30
num_warm_epochs = 5
last_only_epochs = [14, 24]
