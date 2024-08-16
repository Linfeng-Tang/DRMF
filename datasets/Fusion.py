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


class Fusion:
    def __init__(self, config, args=None):
        self.config = config
        self.args = args
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='Restoration'):
        print("=> Utilizing the RestorationDataset_IR() for data loading...")
        train_dataset = RestorationDataset_IR(dir=os.path.join(self.config.data.data_dir, 'train'),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches,
                                        phase='train')
        val_dataset = RestorationDataset_IR(dir=os.path.join(self.config.data.data_dir, 'val'),
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.image_size,
                                      transforms=self.transforms,
                                      parse_patches=parse_patches,
                                      phase='val')

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
    
    def get_fusion_loaders(self, parse_patches=True, validation='Restoration'):
        print("=> Utilizing the FusionDataset() for data loading...")
        train_dataset = FusionDataset(dir=os.path.join(self.args.data_dir, 'train'),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches,
                                        phase='train',
                                        mask=self.config.data.mask,
                                        with_edge = self.config.data.edge, 
                                        fusion_type=self.args.fusion_type)
        val_dataset = FusionDataset(dir=os.path.join(self.args.data_dir, 'val'),
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.image_size,
                                      transforms=self.transforms,
                                      parse_patches=parse_patches,
                                      phase='val',
                                      mask=False,
                                      with_edge = False, 
                                      fusion_type=self.args.fusion_type)

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
    
    
    def get_val_loaders(self, parse_patches=True, phase='test', data_dir=None, data_type=None):    
        if phase == 'fusion':
            print("=> Utilizing the FusionDataset() for data loading...")  
            if data_dir is None:
                data_dir = os.path.join(self.config.data.data_dir, 'val')
            else:
                data_dir = os.path.join(data_dir, 'val')
            val_dataset = FusionDataset(dir=data_dir,
                                    n=self.config.training.patch_n,
                                    patch_size=self.config.data.image_size,
                                    transforms=self.transforms,
                                    parse_patches=parse_patches,
                                    phase='val',
                                    mask=False, 
                                    fusion_type=self.args.fusion_type)
        else:
            print("=> Utilizing the RestorationDataset() for data loading...")   
            if data_dir is None:
                data_dir = os.path.join(self.args.data_dir, phase)
            else:
                data_dir = os.path.join(data_dir, phase)
            val_dataset = RestorationDataset_IR(dir=data_dir,
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches,
                                        phase=phase, 
                                        fusion_type=self.args.fusion_type)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return val_loader
    
    def get_test_loaders(self, parse_patches=True, phase='test', data_dir=None, data_type=None):
        if phase == 'fusion':            
            print("=> Utilizing the FusionDataset() for data loading...")        
            if data_dir is None:
                data_dir = os.path.join(self.config.data.data_dir)
            else:
                data_dir = os.path.join(data_dir)
            test_dataset = FusionDataset(dir=data_dir,
                                    n=self.config.training.patch_n,
                                    patch_size=self.config.data.image_size,
                                    transforms=self.transforms,
                                    parse_patches=parse_patches,
                                    phase='val',
                                    mask=False, 
                                    fusion_type=self.args.fusion_type)
        else:            
            print("=> Utilizing the RestorationDataset() for data loading...")        
            if data_dir is None:
                data_dir = os.path.join(self.args.data_dir, phase)
            else:
                data_dir = os.path.join(data_dir, phase)
            test_dataset = RestorationDataset_IR(dir=data_dir,
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches,
                                        phase=phase, 
                                        fusion_type=self.args.fusion_type)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return test_loader

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True, phase='val', mask=False, with_edge=False, fusion_type="IVIF"):
        super().__init__()
        print('source dir: ', dir)
        source_dir = dir
        A_names, B_names, A_gt_names, B_gt_names= [], [], [], []

        # Raindrop train filelist
        if fusion_type == "IVIF":
            dir_A = os.path.join(source_dir, 'ir_degraded')
            if not os.path.exists(dir_A):                
                dir_A = os.path.join(source_dir, 'ir')
            dir_gt_A = os.path.join(source_dir, 'ir')
            dir_B = os.path.join(source_dir, 'vi_degraded')
            if not os.path.exists(dir_B):                
                dir_B = os.path.join(source_dir, 'vi')
            dir_gt_B = os.path.join(source_dir, 'vi')
        if fusion_type == "MIF":
            dir_A = os.path.join(source_dir, 'CT_degraded')
            if not os.path.exists(dir_A):                
                dir_A = os.path.join(source_dir, 'CT')
            dir_gt_A = os.path.join(source_dir, 'CT')
            dir_B = os.path.join(source_dir, 'MRI_degraded')
            if not os.path.exists(dir_B):                
                dir_B = os.path.join(source_dir, 'MRI')
            dir_gt_B = os.path.join(source_dir, 'MRI')
        self.mask = mask
        self.with_edge = with_edge
        if mask:
            dir_mask = os.path.join(source_dir, 'mask')
            mask_names=[]
        if with_edge:
            dir_edge= os.path.join(source_dir, 'edge')
            edge_names=[]
        
        file_list = natsorted(os.listdir(dir_gt_B))
        for item in file_list:
            if item.endswith('.jpg') or item.endswith('.png') or item.endswith('.bmp'):
                A_names.append(os.path.join(dir_A, item))
                A_gt_names.append(os.path.join(dir_gt_A, item))
                B_names.append(os.path.join(dir_B, item))
                B_gt_names.append(os.path.join(dir_gt_B, item))
                if self.mask:
                    mask_names.append(os.path.join(dir_mask, item))
                if self.with_edge:
                    edge_names.append(os.path.join(dir_edge, item))

        print(len(A_gt_names))
        if phase == 'train':
            x = list(enumerate(A_names))
            random.shuffle(x)
            indices, A_names = zip(*x)
            B_names = [B_names[idx] for idx in indices]
            A_gt_names = [A_gt_names[idx] for idx in indices]        
            B_gt_names = [B_gt_names[idx] for idx in indices]
            if self.mask:
                mask_names = [mask_names[idx] for idx in indices]
            if self.with_edge:
                edge_names = [edge_names[idx] for idx in indices]
            self.dir = None
        if self.mask:
            self.mask_names = mask_names
        if self.with_edge:
            self.edge_names = edge_names
        self.A_names = A_names
        self.B_names = B_names
        self.A_gt_names = A_gt_names
        self.B_gt_names = B_gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        self.phase = phase

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
        A_name = self.A_names[index]
        B_name = self.B_names[index]
        A_gt_name = self.A_gt_names[index]
        B_gt_name = self.B_gt_names[index]
        img_id = re.split('/', A_name)[-1]
        A_img = PIL.Image.open(A_name).convert('RGB')
        B_img = PIL.Image.open(B_name).convert('RGB')
        A_gt_img = PIL.Image.open(A_gt_name).convert('RGB')
        B_gt_img = PIL.Image.open(B_gt_name).convert('RGB')
        if self.mask:
            mask_name = self.mask_names[index]
            mask_img = PIL.Image.open(mask_name).convert('L')
        if self.with_edge:
            edge_name = self.edge_names[index]
            edge_img = PIL.Image.open(edge_name).convert('L')
            
        if self.phase == 'train':
            if self.parse_patches:
                i, j, h, w = self.get_params(A_img, (self.patch_size, self.patch_size), self.n)
                A_img = self.n_random_crops(A_img, i, j, h, w)
                B_img = self.n_random_crops(B_img, i, j, h, w)
                A_gt_img = self.n_random_crops(A_gt_img, i, j, h, w)
                B_gt_img = self.n_random_crops(B_gt_img, i, j, h, w)
                outputs = [torch.cat([self.transforms(A_img[i]), self.transforms(B_img[i]), self.transforms(A_gt_img[i]), self.transforms(B_gt_img[i])], dim=0) for i in range(self.n)]
                if self.mask:
                    mask_img = self.n_random_crops(mask_img, i, j, h, w)
                    outputs = [torch.cat([self.transforms(A_img[i]), self.transforms(B_img[i]), self.transforms(A_gt_img[i]), self.transforms(B_gt_img[i]), self.transforms(mask_img[i])], dim=0) for i in range(self.n)]
                if self.with_edge:
                    edge_img = self.n_random_crops(edge_img, i, j, h, w)
                    outputs = [torch.cat([self.transforms(A_img[i]), self.transforms(B_img[i]), self.transforms(A_gt_img[i]), self.transforms(B_gt_img[i]), self.transforms(mask_img[i]), self.transforms(edge_img[i])], dim=0) for i in range(self.n)]
                return torch.stack(outputs, dim=0), img_id
                return torch.cat([self.transforms(A_img), self.transforms(B_img), self.transforms(A_gt_img), self.transforms(B_gt_img)], dim=0), img_id
        else:
            outputs = torch.cat([self.transforms(A_img), self.transforms(B_img), self.transforms(A_gt_img), self.transforms(B_gt_img)], dim=0)
            if self.mask:                
                # mask_img = mask_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
                outputs =  torch.cat([self.transforms(A_img), self.transforms(B_img), self.transforms(A_gt_img), self.transforms(B_gt_img), self.transforms(mask_img)], dim=0)
            if self.with_edge:                
                # edge_img = edge_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
                outputs =  torch.cat([self.transforms(A_img), self.transforms(B_img), self.transforms(A_gt_img), self.transforms(B_gt_img), self.transforms(edge_img)], dim=0)
            return outputs, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.A_gt_names)


class RestorationDataset_IR(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True, phase='train'):
        super().__init__()
        self.phase = phase
        print('source dir: ', dir)
        source_dir = dir
        deg_names, gt_names = [], []

        # Raindrop train filelist
        dir_deg = os.path.join(source_dir, 'ir_noise')
        dir_gt = os.path.join(source_dir, 'ir')
        
        file_list = natsorted(os.listdir(dir_gt))
        for item in file_list:
            if item.endswith('.jpg') or item.endswith('.png') or item.endswith('.bmp'):
                deg_names.append(os.path.join(dir_deg, item))
                gt_names.append(os.path.join(dir_gt, item))
                
        x = list(enumerate(deg_names))
        if self.phase == 'train' or self.phase == 'val':
            x = list(enumerate(deg_names))
            random.shuffle(x)
            indices, deg_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
        self.dir = None
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
                # Resizing images to multiples of 16 for whole-image restoration
                wd_new, ht_new = deg_img.size
                if ht_new > wd_new and ht_new > 1024:
                    wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                    ht_new = 1024
                elif ht_new <= wd_new and wd_new > 1024:
                    ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                    wd_new = 1024
                wd_new = int(16 * np.ceil(wd_new / 16.0))
                ht_new = int(16 * np.ceil(ht_new / 16.0))  
                deg_img = deg_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
                gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS) 
                return torch.cat([self.transforms(deg_img), self.transforms(gt_img)], dim=0), img_id
        else:
            wd_new, ht_new = deg_img.size
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))  
            deg_img = deg_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS) 
            return torch.cat([self.transforms(deg_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.gt_names)

