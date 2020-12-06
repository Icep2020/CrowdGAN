import os.path
import torchvision.transforms as standard_transforms
import data.transforms as own_transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_png_dataset, make_dmap_dataset
from PIL import Image
import glob
import random
import numpy as np
import torch
import glob
import numbers

class FudanDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # num of sequence frames
        self.max_t_step = opt.max_t_step
        self.min_t_step = opt.min_t_step
        assert self.min_t_step < self.max_t_step
        self.tG = opt.n_frames_G

        self.den_paths = []
        self.img_paths = []
        scenes = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 36, 37, 38, 39, 40, 46, 47, 48, 49, 50, 61, 62, 63, 64, 65,
                  66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 91, 92, 93, 94, 95]
        scene_dirs = [os.path.join(self.root, str(scene)) for scene in scenes]
        #scene_dirs = [os.path.join(self.root, scene) for scene in os.listdir(self.root)]
        for scene_dir in scene_dirs:
            den_paths = sorted(make_dmap_dataset(scene_dir))
            img_paths = sorted(make_png_dataset(scene_dir))
            assert(len(den_paths) == len(img_paths))
            for t_step in range(self.min_t_step, self.max_t_step+1):
                offset = (self.tG - 1) * t_step
                for i in range(len(img_paths) - offset):
                    imgs = [img_paths[i+j*t_step] for j in range(self.tG)]
                    self.img_paths.append(imgs)
                    dens = [den_paths[i+j*t_step] for j in range(self.tG)]
                    self.den_paths.append(dens)
        assert(len(self.img_paths) == len(self.den_paths))

        # train length
        self.num_samples = len(self.img_paths)

        # size
        self.cropSize = (500, 400)
        self.fineSize = (320, 256)
        self.scaling_rate = 1.0 * self.cropSize[0] * self.cropSize[1] / self.fineSize[0] / self.fineSize[1]

        # transforms
        transforms = [own_transforms.RandomCrop_seq(self.cropSize),
                      own_transforms.Scale_seq(self.fineSize)]
        self.main_transform = own_transforms
        transforms.append(own_transforms.RandomHorizontallyFlip_seq())
        self.main_transform = own_transforms.Compose_seq(transforms)
        self.img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dmap_transform = standard_transforms.Compose([
            own_transforms.LabelToTensor(),
            own_transforms.LabelNormalize(opt.logPara)
        ])

    def __getitem__(self, index):
        sequence = []
        dmap_path = img_path = ''
        for i in range(self.tG):
            img_path = self.img_paths[index][i]
            img = Image.open(img_path)
            sequence.append(img)

            dmap_path = self.den_paths[index][i]
            dmap = Image.fromarray(np.load(dmap_path).astype(np.float32, copy=False))
            sequence.append(dmap)

        sequence = self.main_transform(sequence)

        imgs = dmaps = 0
        for i,item in enumerate(sequence):
            if i % 2 == 0:
                img = self.img_transform(item)
                imgs = img if i == 0 else torch.cat([imgs, img], dim=0)
            else:
                dmap = self.dmap_transform(item)
                dmap = torch.ones([self.opt.BP_input_nc, self.fineSize[1], self.fineSize[0]]) * dmap * self.scaling_rate
                dmaps = dmap if i == 1 else torch.cat([dmaps, dmap], dim=0)

        return {'D': dmaps, 'I': imgs, 'D_path': dmap_path, 'I_path': img_path}

    def __len__(self):
        return self.num_samples

    def name(self):
        return 'FudanDataset'



class TestFetcher():
    def __init__(self, __dir_name, n_frames_G, logPara):
        super(TestFetcher, self).__init__()
        self.dmap_paths = glob.glob(os.path.join(__dir_name, 'dmaps/*.npy'))
        self.dmap_paths.sort()
        self.img_paths = glob.glob(os.path.join(__dir_name, 'imgs/*'))
        self.img_paths.sort()
        self.n_frames_G = n_frames_G
        assert(len(self.img_paths) == self.n_frames_G - 1)
        assert(len(self.dmap_paths) > self.n_frames_G - 1)
        self.results_dir = os.path.join(__dir_name, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        try:
            for path in self.img_paths:
                new_path = os.path.join(self.results_dir, path.split('/')[-1])
                os.popen('cp '+ path + ' ' + new_path)
        finally:
            pass
        self.index = 0
        self.img_transform = standard_transforms.Compose([
                    standard_transforms.ToTensor(),
                    standard_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        self.dmap_transform = standard_transforms.Compose([
                    own_transforms.LabelToTensor(),
                    own_transforms.LabelNormalize(logPara)
                ])

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.dmap_paths) - len(self.img_paths):
            raise StopIteration()
        img_paths = [os.path.join(self.results_dir, path) for path in os.listdir(self.results_dir)]
        img_paths.sort()
        img_paths = img_paths[self.index:self.index+self.n_frames_G-1]
        print(img_paths)
        dmap_paths = self.dmap_paths[self.index: self.index+self.n_frames_G]
        imgs = dmaps = 0
        for i,img_path in enumerate(img_paths):
            img = Image.open(img_path)
            img = self.img_transform(img)
            imgs = img if i == 0 else torch.cat([imgs, img], dim=0)
            if i == len(img_paths) - 1:
                imgs = torch.cat([imgs, img], dim=0)

        for i,dmap_path in enumerate(dmap_paths):
            dmap = Image.fromarray(np.load(dmap_path).astype(np.float32, copy=False))
            dmap = self.dmap_transform(dmap)
            dmaps = dmap if i == 0 else torch.cat([dmaps, dmap], dim=0)
        imgs = imgs.unsqueeze(0)
        dmaps = dmaps.unsqueeze(0)
        self.index += 1
        return {'D': dmaps, 'I': imgs, 'D_path': dmap_paths[-1], 'I_path': img_paths[-1]}

    def __len__(self):
        return len(self.dmap_paths) - len(self.img_paths)

