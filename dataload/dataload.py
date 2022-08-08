import os
import random
import numpy as np
import cv2
from scipy import misc, ndimage
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from . import BaseDataLoad

class DataLoad(BaseDataLoad):
    def __init__(self, args, mode):
        super(DataLoad, self).__init__(args, mode)
        self.args = args
        self.mode = mode

        assert(mode == 'train' or mode == 'val' or mode == 'test')

        height, width = (288, 384)
        crop_size = (args.patch_height, args.patch_width)

        self.height = height
        self.width = width
        self.crop_size = crop_size

        self.augment = self.args.augment

        data_dir = self.args.data_dir
        data_list = os.path.join(self.args.data_dir, self.args.data_list)

        self.sample_list = []
        with open(data_list, 'r') as f:
            lines = f.readlines()
            for l in lines:
                paths = l.rstrip().split()
                self.sample_list.append({'rgb': os.path.join(data_dir, paths[0]), 'depth': os.path.join(data_dir, paths[1])})

        if self.mode == 'train':
            self.sample_list = self.sample_list[2000:]
        elif self.mode == 'val':
            self.sample_list = self.sample_list[:2000]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        rgb = Image.open(self.sample_list[idx]['rgb']).convert('RGB')
        dep = cv2.imread(self.sample_list[idx]['depth'], cv2.IMREAD_ANYDEPTH)
        dep = dep.astype(np.float32)
        dep[dep>self.args.max_depth] = 0
        dep = Image.fromarray(dep)

        if self.mode == 'test':
            t_rgb = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep_sp = t_dep(dep)

            output = {'rgb': rgb, 'dep': dep_sp}
            return output

        if self.augment and self.mode == 'train':
            _scale = 1.0
            scale = np.int(self.height * _scale)
            angle = (np.random.rand()-0.5)*70

            rgb = TF.rotate(rgb, angle)
            dep = TF.rotate(dep, angle)

            hfilp = np.random.uniform(0.0, 1.0)
            vfilp = np.random.uniform(0.0, 1.0)
            if hfilp > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
            if vfilp > 0.5:
                rgb = TF.vflip(rgb)
                dep = TF.vflip(dep)

            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale, Image.NEAREST),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            _, height_nocrop, width_nocrop = rgb.shape
            top = np.random.randint(0, height_nocrop - self.crop_size[0])
            left = np.random.randint(0, width_nocrop - self.crop_size[1])
            rgb = rgb[:, top:(top+self.crop_size[0]), left:(left+self.crop_size[1])]
            dep = dep[:, top:(top+self.crop_size[0]), left:(left+self.crop_size[1])]

            dep = dep / _scale

        elif self.mode == 'val':
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

        rgb_n = np.random.uniform(0.0, 1.0)
        if rgb_n > 0.2 and self.args.rgb_noise > 0:
            rgb_noise = torch.normal(mean=torch.zeros_like(rgb), std=self.args.rgb_noise * np.random.uniform(0.5, 1.5))
            rgb = rgb + rgb_noise

        
        if self.args.noise:
            reflection = np.clip(np.random.normal(1, scale=0.333332, size=(1,1)), 0.01, 3)[0,0]
            noise = torch.normal(mean=0.0, std=dep * reflection * self.args.noise)
            dep_noise = dep + noise
            dep_noise[dep_noise < 0] = 0
        else:
            dep_noise = dep.clone()

        if self.args.grid_spot:
            dep_sp = self.get_sparse_depth_grid(dep_noise)
        else:
            dep_sp = self.get_sparse_depth(dep_noise, self.args.num_spot)

        if self.args.cut_mask:
            dep_sp = self.cut_mask(dep_sp)


        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep}

        return output

    def get_sparse_depth(self, dep, num_spot):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_spot]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp

    def get_sparse_depth_grid(self, dep):
        '''
        Simulate pincushion distortion:
        --stride: 
        It controls the distance between neighbor spots7
        Suggest stride value:       5~10

        --dist_coef:
        It controls the curvature of the spot pattern
        Larger dist_coef distorts the pattern more.
        Suggest dist_coef value:    0 ~ 5e-5

        --noise:
        standard deviation of the spot shift
        Suggest noise value:        0 ~ 0.5
        '''

        # Generate Grid points
        channel, img_h, img_w = dep.shape
        assert channel == 1

        stride = np.random.randint(5,7)

        dist_coef = np.random.rand()*4e-5 + 1e-5
        noise = np.random.rand() * 0.3

        x_odd, y_odd = np.meshgrid(np.arange(stride//2, img_h, stride*2), np.arange(stride//2, img_w, stride))
        x_even, y_even = np.meshgrid(np.arange(stride//2+stride, img_h, stride*2), np.arange(stride, img_w, stride))
        x_u = np.concatenate((x_odd.ravel(),x_even.ravel()))
        y_u = np.concatenate((y_odd.ravel(),y_even.ravel()))
        x_c = img_h//2 + np.random.rand()*50-25
        y_c = img_w//2 + np.random.rand()*50-25
        x_u = x_u - x_c
        y_u = y_u - y_c       
        

        # Distortion
        r_u = np.sqrt(x_u**2+y_u**2)
        r_d = r_u + dist_coef * r_u**3
        num_d = r_d.size
        sin_theta = x_u/r_u
        cos_theta = y_u/r_u
        x_d = np.round(r_d * sin_theta + x_c + np.random.normal(0, noise, num_d))
        y_d = np.round(r_d * cos_theta + y_c + np.random.normal(0, noise, num_d))
        idx_mask = (x_d<img_h) & (x_d>0) & (y_d<img_w) & (y_d>0)
        x_d = x_d[idx_mask].astype('int')
        y_d = y_d[idx_mask].astype('int')

        spot_mask = np.zeros((img_h, img_w))
        spot_mask[x_d,y_d] = 1

        dep_sp = torch.zeros_like(dep)
        dep_sp[:, x_d, y_d] = dep[:, x_d, y_d]

        return dep_sp

    def cut_mask(self, dep):
        _, h, w = dep.size()
        c_x = np.random.randint(h/4, h/4*3)
        c_y = np.random.randint(w/4, w/4*3)
        r_x = np.random.randint(h/4, h/4*3)
        r_y = np.random.randint(h/4, h/4*3)

        mask = torch.zeros_like(dep)
        min_x = max(c_x-r_x, 0)
        max_x = min(c_x+r_x, h)
        min_y = max(c_y-r_y, 0)
        max_y = min(c_y+r_y, w)
        mask[0, min_x:max_x, min_y:max_y] = 1
        
        return dep * mask
        