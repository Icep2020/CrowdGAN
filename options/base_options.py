import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='../all-things-CrowdGAN/video_dataset/fudan', help='path to images (should have subfolders including scenes)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--cropSize', type=int, default=512, help='crop images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--logPara', type=int, default=100, help='density map scaling rate')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--random_seed', default=1000, type=int, help='random seed number')
        self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--model', type=str, default='Final',help='chooses which model to use: STG | PFP | Final')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')

        self.parser.add_argument('--P_input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--BP_input_nc', type=int, default=1, help='# of input feature channels')
        self.parser.add_argument('--padding_type', type=str, default='reflect', help='# of input image channels')

        # for pretrained model path
        self.parser.add_argument('--flownet_ckpt', type=str, default='', help='path of flownet pretrained model')
        self.parser.add_argument('--flowG_ckpt', type=str, default='', help='path of flow generator pretrained model')
        self.parser.add_argument('--mapG_ckpt', type=str, default='', help='path of map generator pretrained model')
        self.parser.add_argument('--netG_ckpt', type=str, default='', help='path of post generator pretrained model for finetune')
        self.parser.add_argument('--netD_PB_ckpt', type=str, default='', help='path of discriminator PB pretrained model for finetune')
        self.parser.add_argument('--netD_PP_ckpt', type=str, default='', help='path of discriminator PP pretrained model for finetune')
        self.parser.add_argument('--netD_T_ckpt', type=str, default='', help='path of discriminator T pretrained model for finetune')

        # for experiment setting
        self.parser.add_argument('--n_frames_G', type=int, default=4, help='number of frames to feed into generator, i.e., n_frames_G-1 is the number of frames we look into past')

        # for discriminators to use
        self.parser.add_argument('--with_D_PP', type=int, default=1, help='use D to judge P and P is pair or not')
        self.parser.add_argument('--with_D_PB', type=int, default=1, help='use D to judge P and B is pair or not')
        self.parser.add_argument('--with_D_T', type=int, default=1, help='use D to judge temporal consistency')

        # down-sampling times
        self.parser.add_argument('--G_n_downsampling', type=int, default=2, help='down-sampling blocks for generator')
        self.parser.add_argument('--D_n_downsampling', type=int, default=2, help='down-sampling blocks for discriminator')
        self.parser.add_argument('--P_n_downsampling', type=int, default=2, help='down-sampling blocks for post generator')

        self.parser.add_argument('--isDropout', action='store_true', help='if specified, use dropout')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

