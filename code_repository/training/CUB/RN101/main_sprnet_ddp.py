import argparse
# import copy
import os
import pickle
import random
import re
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

import model
# import prune
# import push
import save
import train_and_test_sprnet_ddp as tnt
from helpers import makedir
from preprocess import mean, std
from settings import (add_on_layers_type, base_architecture, coefs,
                      dataset_path, exp_name, img_size, joint_optimizer_lrs,
                      num_classes, num_train_epochs, num_warm_epochs,
                      prototype_activation_function, prototype_shape,
                      test_batch_size, train_batch_size, warm_optimizer_lrs)

RANDOM_SEED = 2023

model_dir = f'./saved_models/{exp_name}'
log_filename_pattern = os.path.join(model_dir, r'train_r{}.log')

weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'


def setup():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    # book keeping namings and code
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, 'latest_checkpoint.pth')):
        print('Model directory already exists. Continue to use latest checkpoint? (y/n)')
        if input() == 'n':
            exit()

    makedir(model_dir)
    backup_files = [
        __file__,
        'settings.py',
        'hook_features.py',
        'model.py',
        'train_and_test_sprnet_ddp.py',
    ]
    for file in backup_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            shutil.copy(src=os.path.join(os.getcwd(), file), dst=model_dir)

    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)


class Cropped_CUB_with_Semantic(datasets.ImageFolder):
    splits = ['test_cropped', 'train_cropped', 'train_cropped_augmented',
              'train_cropped_augmented_ex', 'train_cropped_augmented_ex2']
    CLIP_CONTEXT_LENGTH = 77  # TODO: this is a hack

    def __init__(self, root, split, **kwargs):
        root = self.root = os.path.expanduser(root)
        assert split in self.splits
        self.split = split
        self.split_folder = os.path.join(root, 'cub200_cropped', split)
        super().__init__(root=self.split_folder, **kwargs)
        self.root = root
        self._init_captions()
        self._init_class_attribute_matrix()
        self._init_class_attribute_tokens()
        self._init_global_attribute_tokens()

    def _init_captions(self):
        caption_dir = os.path.join(self.root, 'captioning/text')
        self.captions = dict()
        for c in os.listdir(caption_dir):
            for f in os.listdir(f'{caption_dir}/{c}'):
                self.captions[os.path.splitext(f)[0]] = []
                with open(f'{caption_dir}/{c}/{f}') as fp:
                    for line in fp:
                        line = line.strip()
                        if line == '':
                            continue
                        self.captions[os.path.splitext(f)[0]].append(line)

    def _init_class_attribute_matrix(self):
        class_attribute_matrix_path = os.path.join(
            self.root, 'class_attribute_matrix.pkl')
        with open(class_attribute_matrix_path, 'rb') as fp:
            self.class_attribute_matrix = pickle.load(fp)

    def _init_class_attribute_tokens(self):
        class_attribute_tokens_path = os.path.join(
            self.root, 'class_attribute_tokens.txt')
        self.class_attribute_tokens = []
        with open(class_attribute_tokens_path) as fp:
            for line in fp:
                line = line.strip()
                line = line.split(', ')
                self.class_attribute_tokens.append(line)

    def _init_global_attribute_tokens(self):
        global_attribute_tokens_path = os.path.join(
            self.root, 'attributes', 'attribute_tokens_0203.txt')
        self.global_attribute_tokens = []
        with open(global_attribute_tokens_path) as fp:
            for line in fp:
                line = line.strip()
                if line == "":
                    continue
                line = line.split('##')
                self.global_attribute_tokens.append(
                    "a photo of one part of a bird, the bird " +
                    line[1]
                )

    def __getitem__(self, index):
        path, target = self.samples[index]

        attribute = torch.tensor(
            self.class_attribute_matrix[target], dtype=torch.float)

        if '_original_' in path:  # augmented
            filename = re.search(r'(?<=original_).*?(?=.JPG)',
                                 os.path.basename(path)).group()
        else:
            filename = os.path.splitext(os.path.basename(path))[0]

        if random.random() < 0.7:  # TODO: change this
            # if True:
            global_str = random.choice(self.captions[filename])[:77]
        else:
            tokens = self.class_attribute_tokens[target]
            global_str = tokens[0]
            att_tokens = sorted(tokens[1:])
            random.shuffle(att_tokens)
            for token in att_tokens:
                temp_str = global_str + ', ' + token
                if len(temp_str) >= self.CLIP_CONTEXT_LENGTH:
                    break
                global_str = temp_str

        sample, target = super().__getitem__(index)
        return sample, target, attribute, global_str


def single_process(rank, world_size):
    with open(log_filename_pattern.format(rank), 'a') as fp_log:
        def log(text):
            formatted_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            text_with_rank = f'[r{rank}][{formatted_time}] {text}'
            print(text_with_rank)
            fp_log.write(text_with_rank + '\n')
            fp_log.flush()

        log('single process start')
        # create default process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # load the data
        torchvision.set_image_backend('accimage')
        normalize = transforms.Normalize(mean=mean, std=std)

        # all datasets
        # train set
        single_train_batch_size = train_batch_size // world_size
        train_dataset = Cropped_CUB_with_Semantic(
            dataset_path,
            split='train_cropped_augmented_ex',
            transform=transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        train_dataset[23000]
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, rank=rank, shuffle=True, seed=RANDOM_SEED)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=single_train_batch_size, num_workers=8, pin_memory=True, sampler=train_sampler)

        # test set
        single_test_batch_size = test_batch_size // world_size
        test_dataset = Cropped_CUB_with_Semantic(
            dataset_path,
            split='test_cropped',
            transform=transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ])
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, rank=rank, shuffle=False, seed=RANDOM_SEED)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=single_test_batch_size, num_workers=8, pin_memory=True, sampler=test_sampler)

        log(f'training set size: {len(train_loader.dataset)}')
        log(f'test set size: {len(test_loader.dataset)}')
        log(f'training batch size: {single_train_batch_size}')
        log(f'test batch size: {single_test_batch_size}')

        # create local model
        sprnet = model.construct_SPRNet(base_architecture=base_architecture,
                                        pretrained=True, img_size=img_size,
                                        prototype_shape=prototype_shape,
                                        num_classes=num_classes,
                                        prototype_activation_function=prototype_activation_function,
                                        add_on_layers_type=add_on_layers_type)
        sprnet = sprnet.to(rank)

        sprnet_ddp = DDP(sprnet, device_ids=[
                         rank], find_unused_parameters=True)
        class_specific = True

        # define optimizer

        joint_optimizer_specs = [
            {'params': sprnet_ddp.module.image_encoder.parameters(),
                'lr': joint_optimizer_lrs['features'],
                # 'weight_decay': 1e-2,
                # for AdamW default weight_decay is 1e-2
             },  # bias are now also being regularized
            {'params': sprnet_ddp.module.add_on_layers.parameters(),
                'lr': joint_optimizer_lrs['add_on_layers'],
                # 'weight_decay': 1e-4,
                # for AdamW default weight_decay is 1e-2
             },
            {'params': sprnet_ddp.module.prototype_vectors,
                'lr': joint_optimizer_lrs['prototype_vectors'],
                # 'weight_decay': 1e-4,
                # for AdamW default weight_decay is 1e-2
             },
            {'params': sprnet_ddp.module.text_encoder.parameters(),
                'lr': joint_optimizer_lrs['clip'],
                # "betas": (0.9, 0.98),
                # "eps": 1e-6,
                # 'weight_decay': 1e-2,
                # for AdamW default weight_decay is 1e-2
             },
        ]
        # joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     joint_optimizer, step_size=joint_lr_step_size, gamma=joint_lr_step_gamma)
        joint_optimizer = torch.optim.AdamW(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            joint_optimizer, T_max=num_train_epochs - num_warm_epochs)

        warm_optimizer_specs = [
            {'params': sprnet_ddp.module.add_on_layers.parameters(),
                'lr': warm_optimizer_lrs['add_on_layers'],
                # 'weight_decay': 1e-4,
                # for AdamW default weight_decay is 1e-2
             },
            {'params': sprnet_ddp.module.prototype_vectors,
                'lr': warm_optimizer_lrs['prototype_vectors'],
                # 'weight_decay': 1e-4,
                # for AdamW default weight_decay is 1e-2
             },
        ]
        # warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
        warm_optimizer = torch.optim.AdamW(warm_optimizer_specs)

        # last_layer_optimizer_specs = [
        #     {'params': sprnet_ddp.module.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        # last_layer_optimizer = torch.optim.AdamW(last_layer_optimizer_specs)

        epoch = 0

        # load training checkpoint
        training_checkpoint_path = os.path.join(
            model_dir, 'latest_checkpoint.pth')
        if os.path.exists(training_checkpoint_path):
            dist.barrier()
            log(f'loading checkpoint from {training_checkpoint_path}')
            map_location = {'cuda:0': f'cuda:{rank}'}
            checkpoint = torch.load(training_checkpoint_path,
                                    map_location=map_location)
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
                log(f'epoch loaded')
            if 'model_state_dict' in checkpoint:
                sprnet_ddp.load_state_dict(checkpoint['model_state_dict'])
                log(f'model_state_dict loaded')
            if 'warm_optimizer_state_dict' in checkpoint:
                warm_optimizer.load_state_dict(
                    checkpoint['warm_optimizer_state_dict'])
                log(f'warm_optimizer_state_dict loaded')
            if 'joint_optimizer_state_dict' in checkpoint:
                joint_optimizer.load_state_dict(
                    checkpoint['joint_optimizer_state_dict'])
                log(f'joint_optimizer_state_dict loaded')
            # if 'last_layer_optimizer_state_dict' in checkpoint:
            #     last_layer_optimizer.load_state_dict(
            #         checkpoint['last_layer_optimizer_state_dict'])
            #     log(f'last_layer_optimizer_state_dict loaded')
            if 'joint_lr_scheduler_state_dict' in checkpoint:
                joint_lr_scheduler.load_state_dict(
                    checkpoint['joint_lr_scheduler_state_dict'])
                log(f'joint_lr_scheduler_state_dict loaded')

        dist.barrier()
        # train the model
        log('start training')

        while epoch < num_train_epochs:
            log(f'epoch: \t{epoch}')
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

            if epoch < num_warm_epochs:
                tnt.warm_only(model=sprnet_ddp, log=log)
                _ = tnt.train(model=sprnet_ddp, dataloader=train_loader, optimizer=warm_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
            else:
                tnt.joint(model=sprnet_ddp, log=log)
                _ = tnt.train(model=sprnet_ddp, dataloader=train_loader, optimizer=joint_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)

                log('\tcurrent lr: ' + repr(joint_lr_scheduler.get_last_lr()))
                joint_lr_scheduler.step()

            # TODO
            accu = tnt.test(model=sprnet_ddp, dataloader=test_loader,
                            class_specific=class_specific, log=log)

            if rank == 0:
                save.save_model_w_condition(model=sprnet_ddp, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                            target_accu=0.7, log=log)
            dist.barrier()

            # if epoch >= num_warm_epochs and epoch in last_only_epochs:
            #     if prototype_activation_function != 'linear':
            #         tnt.last_only(model=sprnet_ddp, log=log)
            #         for i in range(10):
            #             log('iteration: \t{0}'.format(i))
            #             _ = tnt.train(model=sprnet_ddp, dataloader=train_loader, optimizer=last_layer_optimizer,
            #                           class_specific=class_specific, coefs=coefs, log=log)
            #             accu = tnt.test(model=sprnet_ddp, dataloader=test_loader,
            #                             class_specific=class_specific, log=log)
            #             if rank == 0:
            #                 save.save_model_w_condition(model=sprnet_ddp, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
            #                                             target_accu=0.70, log=log)
            #             dist.barrier()

            epoch += 1

            # save training checkpoint
            epoch_training_checkpoint_path = training_checkpoint_path.replace(
                'latest_checkpoint.pth', f'checkpoint_{epoch}.pth')
            if rank == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': sprnet_ddp.state_dict(),
                    'warm_optimizer_state_dict': warm_optimizer.state_dict(),
                    'joint_optimizer_state_dict': joint_optimizer.state_dict(),
                    # 'last_layer_optimizer_state_dict': last_layer_optimizer.state_dict(),
                    'joint_lr_scheduler_state_dict': joint_lr_scheduler.state_dict(),
                }
                torch.save(obj=checkpoint, f=epoch_training_checkpoint_path)

                if os.path.exists(training_checkpoint_path):
                    os.unlink(training_checkpoint_path)
                # use relative path in symlink
                os.symlink(f'checkpoint_{epoch}.pth', training_checkpoint_path)

            dist.barrier()
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(training_checkpoint_path,
                                    map_location=map_location)
            sprnet_ddp.load_state_dict(checkpoint['model_state_dict'])
            # TODO: any optimizer required to load?


def main():
    # TODO: single-machine multi-GPU now
    world_size = torch.cuda.device_count()
    mp.spawn(single_process,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = f'{26371 + int(exp_name[-4:])}'

    setup()
    main()
