exp_name = 'cub-vitb16-0000'

base_architecture = 'clip_vitb16'
img_size = 224
prototype_shape = (2000, 768, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

dataset_path = './datasets/CUB'
train_batch_size = 80
test_batch_size = 160

joint_optimizer_lrs = {'features': 5e-6,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3,
                       'clip': 5e-6}

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'global_clip': 0.1,
    'local_clip': 0.1,
}

num_train_epochs = 50
num_warm_epochs = 10
