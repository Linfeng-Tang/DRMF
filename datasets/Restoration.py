import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
from natsort import natsorted


class Restoration:
    def __init__(self, config, args=None):
        self.config = config
        self.args = args
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='Restoration'):
        print("=> Utilizing the RestorationDataset() for data loading...")
        train_dataset = RestorationDataset(dir=os.path.join(self.args.data_dir, 'train'),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches,
                                        phase='train',
                                        data_type=self.args.data_type)
        
        val_dataset = RestorationDataset(dir=os.path.join(self.args.data_dir, 'val'),
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.image_size,
                                      transforms=self.transforms,
                                      parse_patches=parse_patches,
                                      phase='val',                                       
                                      data_type=self.args.data_type)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
    
    def get_test_loaders(self, parse_patches=True, phase='test', data_dir=None):
        print("=> Utilizing the RestorationDataset() for data loading...")
        if data_dir is None:
            data_dir = os.path.join(self.args.data_dir)
        else:
            data_dir = os.path.join(data_dir)
        test_dataset = RestorationDataset(dir=data_dir,
                                    n=self.config.training.patch_n,
                                    patch_size=self.config.data.image_size,
                                    transforms=self.transforms,
                                    parse_patches=parse_patches,
                                    phase=phase,
                                    data_type=self.args.data_type)
        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return test_loader

class RestorationDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True, phase='train', data_type='LOL'):
        super().__init__()
        print('source dir: ', dir)
        self.phase = phase
        source_dir = dir
        deg_names, gt_names = [], []
        if self.phase == 'train':
            if data_type == 'LOL':
                dir_deg = os.path.join(source_dir, 'low')
                dir_gt = os.path.join(source_dir, 'high')
            elif data_type == 'IR':            
                dir_deg = os.path.join(source_dir, 'ir_degraded')
                dir_gt = os.path.join(source_dir, 'ir')
            elif data_type == 'VI':            
                dir_deg = os.path.join(source_dir, 'vi_degraded')
                dir_gt = os.path.join(source_dir, 'vi')
            elif data_type == 'CT':            
                dir_deg = os.path.join(source_dir, 'CT_degraded')
                dir_gt = os.path.join(source_dir, 'CT')
            elif data_type == 'MRI':            
                dir_deg = os.path.join(source_dir, 'MRI_degraded')
                dir_gt = os.path.join(source_dir, 'MRI')        
            elif data_type == 'CT_norm':            
                dir_deg = os.path.join(source_dir, 'CT')
                dir_gt = os.path.join(source_dir, 'CT')
            elif data_type == 'MRI_norm':            
                dir_deg = os.path.join(source_dir, 'MRI')
                dir_gt = os.path.join(source_dir, 'MRI')
            else:
                dir_deg = os.path.join(source_dir, 'low')
                dir_gt = os.path.join(source_dir, 'high')            
        else:
            if data_type == 'LOL':
                dir_deg = os.path.join(source_dir, 'low')
                dir_gt = os.path.join(source_dir, 'high')
            elif data_type == 'IR':            
                dir_deg = os.path.join(source_dir, 'ir_degraded')
                if not os.path.exists(dir_deg):
                    dir_deg = os.path.join(source_dir, 'ir')
                dir_gt = os.path.join(source_dir, 'ir')
            elif data_type == 'VI':            
                dir_deg = os.path.join(source_dir, 'vi_degraded')
                if not os.path.exists(dir_deg):
                    dir_deg = os.path.join(source_dir, 'vi')
                dir_gt = os.path.join(source_dir, 'vi')
            elif data_type == 'CT':            
                dir_deg = os.path.join(source_dir, 'CT_degraded')
                if not os.path.exists(dir_deg):
                    dir_deg = os.path.join(source_dir, 'CT')
                dir_gt = os.path.join(source_dir, 'CT')
            elif data_type == 'MRI':            
                dir_deg = os.path.join(source_dir, 'MRI_degraded')
                if not os.path.exists(dir_deg):
                    dir_deg = os.path.join(source_dir, 'MRI')
                dir_gt = os.path.join(source_dir, 'MRI')        
            elif data_type == 'CT_norm':            
                dir_deg = os.path.join(source_dir, 'CT')
                dir_gt = os.path.join(source_dir, 'CT')
            elif data_type == 'MRI_norm':            
                dir_deg = os.path.join(source_dir, 'MRI')
                dir_gt = os.path.join(source_dir, 'MRI')
            else:
                dir_deg = os.path.join(source_dir, 'low')
                dir_gt = os.path.join(source_dir, 'high') 
        print("Degreded floder: {}, Reference folder: {}".format(dir_deg, dir_gt))
        deg_names, gt_names = [], []
        file_list = natsorted(os.listdir(dir_gt))
        for item in file_list:                
            if item.endswith('.jpg') or item.endswith('.png') or item.endswith('.bmp'):
                deg_names.append(os.path.join(dir_deg, item))
                gt_names.append(os.path.join(dir_gt, item))
        print("The number of the training dataset is: {}".format(len(gt_names)))
        
        if self.phase == 'train':
            x = list(enumerate(deg_names))
            random.shuffle(x)
            indices, deg_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
        
        self.dir = None        
        print("The number of the testing dataset is: {}".format(len(gt_names)))
        self.deg_names = deg_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)] 
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        deg_name = self.deg_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', deg_name)[-1]
        deg_img = PIL.Image.open(deg_name).convert('RGB')
        gt_img = PIL.Image.open(gt_name).convert('RGB')
        if self.phase == 'train':
            if self.parse_patches:
                i, j, h, w = self.get_params(deg_img, (self.patch_size, self.patch_size), self.n)
                deg_img = self.n_random_crops(deg_img, i, j, h, w)
                gt_img = self.n_random_crops(gt_img, i, j, h, w)
                outputs = [torch.cat([self.transforms(deg_img[i]), self.transforms(gt_img[i])], dim=0) for i in range(self.n)]
                return torch.stack(outputs, dim=0), img_id
            else:
                return torch.cat([self.transforms(deg_img), self.transforms(gt_img)], dim=0), img_id
        else:
            return torch.cat([self.transforms(deg_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.gt_names)