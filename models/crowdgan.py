import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .flow_predict.FlowSD import *

class CrowdganModel(BaseModel):
    def name(self):
        return 'CrowdganModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.log_para = opt.logPara
        self.tG = opt.n_frames_G
        self.output_nc = opt.output_nc
        self.P_input_nc = opt.P_input_nc
        self.BP_input_nc = opt.BP_input_nc

        flowG_input_nc = [opt.P_input_nc + opt.BP_input_nc*(opt.n_frames_G) +  2*(opt.n_frames_G-2)]
        mapG_input_nc = [opt.P_input_nc, opt.BP_input_nc*2, opt.P_input_nc*(opt.n_frames_G-1)]
        fusion_input_nc = [opt.ngf + opt.ngf]
        n_layers_flowG = [6]
        n_layers_mapG = [4,4]
        n_layers_postG = [2]
        if self.isTrain:
            self.tD = opt.n_frames_D
            n_layers_D_PB = 3
            n_layers_D_PP = 3
            n_layers_D_T = 3
            netD_PB_input_nc = opt.output_nc + opt.BP_input_nc
            netD_PP_input_nc = opt.output_nc + opt.output_nc
            netD_T_input_nc = opt.output_nc * opt.n_frames_D

        self.mapG = networks.define_G(mapG_input_nc, self.output_nc,
                                     opt.ngf, 'Transfer', n_layers_mapG,
                                     opt.norm, opt.init_type, self.gpu_ids,
                                     n_downsampling=opt.G_n_downsampling, use_dropout=opt.isDropout, fusion_stage=True)
        self.mapG.load_state_dict(torch.load(opt.mapG_ckpt), strict=False)

        self.flowNet = FlowSD()
        self.flowNet.load_state_dict(torch.load(opt.flownet_ckpt))
        self.flowNet.eval()
        self.flowNet = torch.nn.DataParallel(self.flowNet, device_ids=self.gpu_ids).cuda()

        self.flowG = networks.define_G(flowG_input_nc, 2,
                                      opt.ngf, 'FlowEst', n_layers_flowG,
                                      opt.norm, opt.init_type, self.gpu_ids,
                                      n_downsampling=opt.G_n_downsampling, fusion_stage=True)
        self.flowG.load_state_dict(torch.load(opt.flowG_ckpt))

        self.netG = networks.define_G(fusion_input_nc, 1,
                                      opt.ngf, 'Fusion', n_layers_postG,
                                      opt.norm, opt.init_type, self.gpu_ids,
                                      n_downsampling=opt.P_n_downsampling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan

            if opt.with_D_PB:
                self.netD_PB = networks.define_D(netD_PB_input_nc, opt.ndf,
                                            'resnet',
                                            n_layers_D_PB, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(netD_PP_input_nc, opt.ndf,
                                            'resnet',
                                            n_layers_D_PP, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            n_downsampling = opt.D_n_downsampling)
            if opt.with_D_T:
                self.netD_T = networks.define_D(netD_T_input_nc, opt.ndf,
                                            'resnet',
                                            n_layers_D_T, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            n_downsampling = opt.D_n_downsampling)
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            self.fake_T_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_mapG = torch.optim.Adam(self.mapG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_flowG = torch.optim.Adam(self.flowG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_T:
                self.optimizer_D_T = torch.optim.Adam(self.netD_T.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_mapG)
            self.optimizers.append(self.optimizer_flowG)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            if opt.with_D_T:
                self.optimizers.append(self.optimizer_D_T)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.mapG)
        networks.print_network(self.flowG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
            if opt.with_D_T:
                networks.print_network(self.netD_T)

        print('-----------------------------------------------')

    def forward(self):
        self.input_prev_I = Variable(self.input_prev_I_set)
        self.input_prev_D = Variable(self.input_prev_D_set)
        self.input_last_I = Variable(self.input_last_I_set)
        self.input_last_D = Variable(self.input_last_D_set)
        self.input_curr_I = Variable(self.input_curr_I_set)
        self.input_curr_D = Variable(self.input_curr_D_set)

        # flowG inference
        b, _, h, w = self.input_curr_I.size()
        input_post_I = torch.cat([self.input_prev_I, self.input_curr_I], dim=1)[:,3:].contiguous().view(-1, 3, h, w)
        input_prev_I = self.input_prev_I.contiguous().view(-1, 3, h, w)
        flow_predict_input = torch.cat([input_prev_I, input_post_I], dim=1)
        flow = self.flowNet(flow_predict_input)
        flow_input = flow.contiguous().view(b, -1, h, w)[:,:-2]
        flowG_input = torch.cat([self.input_last_I, self.input_prev_D, self.input_curr_D, flow_input.detach()], dim=1)
        flow_output = self.flowG(flowG_input)
        flow_predict = flow_output['out']
        flow_feature = flow_output['fea']
        self.warp = self.resample(self.input_last_I, flow_predict)

        # mapG inference
        mapG_input = [self.input_last_I, torch.cat((self.input_last_D, self.input_curr_D), dim=1), self.input_prev_I]
        map_output = self.mapG(mapG_input)
        self.res = map_output['out']
        map_feature = map_output['fea']

        # netG inference
        G_input = [map_feature, flow_feature]
        weight = self.netG(G_input)
        self.fake = self.res * weight + self.warp * (1 - weight)


    def backward_G(self):

        # GAN loss
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake, self.input_curr_D), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake, self.input_last_I), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        if self.opt.with_D_T:
            pred_fake_T = self.netD_T(torch.cat((self.input_prev_I, self.fake), 1))
            self.loss_G_GAN_T = self.criterionGAN(pred_fake_T, True)

        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN

        if self.opt.with_D_T:
            temporal_GANloss = self.loss_G_GAN_T * self.opt.lambda_GAN_T

        # L1 loss
        self.loss_G_L1 = self.criterionL1(self.fake, self.input_curr_I) * self.opt.lambda_L1

        pair_L1loss = self.loss_G_L1
        pair_loss = pair_L1loss
        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss += pair_GANloss
        if self.opt.with_D_T:
            pair_loss += temporal_GANloss

        pair_loss.backward()

        self.pair_L1loss = pair_L1loss.data
        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.data
        if self.opt.with_D_T:
            self.temporal_GANloss = temporal_GANloss.data

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_curr_I, self.input_curr_D), 1)
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake, self.input_curr_D), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)
        self.loss_D_PB = loss_D_PB.data

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_curr_I, self.input_last_I), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake, self.input_last_I), 1).data )
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)
        self.loss_D_PP = loss_D_PP.data

    # D: take(prev, P`, flows) as input
    def backward_D_T(self):
        real_T = torch.cat((self.input_prev_I, self.input_curr_I), 1)
        fake_T = self.fake_T_pool.query(torch.cat((self.input_prev_I, self.fake), 1).data)
        loss_D_T = self.backward_D_basic(self.netD_T, real_T, fake_T)
        self.loss_D_T = loss_D_T.data

    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.optimizer_mapG.zero_grad()
        self.optimizer_flowG.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_mapG.step()
        self.optimizer_flowG.step()

        # D_PP
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP()
                self.optimizer_D_PP.step()
        # D_BP
        if self.opt.with_D_PB:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB()
                self.optimizer_D_PB.step()
        # D_T
        if self.opt.with_D_T:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_T.zero_grad()
                self.backward_D_T()
                self.optimizer_D_T.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('pair_L1loss', self.pair_L1loss)])
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_PB or self.opt.with_D_PP:
            ret_errors['pair_GANloss'] = self.pair_GANloss
        if self.opt.with_D_T:
            ret_errors['temporal_GANloss'] = self.temporal_GANloss
        return ret_errors


    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)
        self.save_network(self.mapG,  'mapG',  label, self.gpu_ids)
        self.save_network(self.flowG, 'flowG', label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB, 'netD_PB', label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)
        if self.opt.with_D_T:
            self.save_network(self.netD_T, 'netD_T', label, self.gpu_ids)
