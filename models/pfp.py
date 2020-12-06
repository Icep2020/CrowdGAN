import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks
from .flow_predict.FlowSD import *


class PFPModel(BaseModel):
    def name(self):
        return 'PFPModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.log_para = opt.logPara
        self.tG = opt.n_frames_G
        self.output_nc = 2
        self.P_input_nc = opt.P_input_nc
        self.BP_input_nc = opt.BP_input_nc
        netG_image_nc = opt.P_input_nc
        netG_dmap_nc = opt.BP_input_nc * (opt.n_frames_G)
        netG_flow_nc = 2 * (opt.n_frames_G-2)
        netG_input_nc = [netG_image_nc+netG_dmap_nc+netG_flow_nc]
        n_layers_G = [6]

        self.netG = networks.define_G(netG_input_nc, self.output_nc,
                                      opt.ngf, 'FlowEst', n_layers_G,
                                      opt.norm, opt.init_type, self.gpu_ids,
                                      n_downsampling=opt.G_n_downsampling)
        self.flowNet = FlowSD()
        self.flowNet.load_state_dict(torch.load(opt.flownet_ckpt))
        self.flowNet.eval()
        self.flowNet = torch.nn.DataParallel(self.flowNet, device_ids=self.gpu_ids).cuda()


        if self.isTrain:
            self.tD = opt.n_frames_D
            self.old_lr = opt.lr

            # define loss functions
            self.criterionFlow = torch.nn.L1Loss()
            self.criterionWarp = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def forward(self):
        self.input_prev_I = Variable(self.input_prev_I_set)
        self.input_prev_D = Variable(self.input_prev_D_set)
        self.input_last_I = Variable(self.input_last_I_set)
        self.input_last_D = Variable(self.input_last_D_set)
        self.input_curr_I = Variable(self.input_curr_I_set)
        self.input_curr_D = Variable(self.input_curr_D_set)
        b, _, h, w = self.input_curr_I.size()
        input_post_I = torch.cat([self.input_prev_I, self.input_curr_I], dim=1)[:,3:].contiguous().view(-1, 3, h, w)
        input_prev_I = self.input_prev_I.contiguous().view(-1, 3, h, w)
        flow_predict_input = torch.cat([input_prev_I, input_post_I], dim=1)
        flow = self.flowNet(flow_predict_input)
        flow_input = flow.contiguous().view(b, -1, h, w)[:,:-2]
        self.flow_gt = flow.contiguous().view(b, -1, h, w)[:,-2:]
        G_input = torch.cat([self.input_last_I, self.input_prev_D, self.input_curr_D, flow_input], dim=1)
        self.flow_predict = self.netG(G_input)
        self.warp = self.resample(self.input_last_I, self.flow_predict)
        self.dmap_warp = self.resample(self.input_last_D, self.flow_predict)


    def backward_G(self):
        # compute reference flow
        flow_loss = self.criterionFlow(self.flow_predict, self.flow_gt) * self.opt.lambda_F
        warp_loss = self.criterionWarp(self.warp, self.input_curr_I) * self.opt.lambda_W
        dmap_warp_loss = self.criterionWarp(self.dmap_warp, self.input_curr_D) * self.opt.lambda_dw
        pair_loss = flow_loss + warp_loss + dmap_warp_loss

        pair_loss.backward()

        self.pair_loss = pair_loss.data
        self.flow_loss = flow_loss.data
        self.warp_loss = warp_loss.data
        self.dmap_warp_loss = dmap_warp_loss.data

    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([ ('pairloss', self.pair_loss)])
        ret_errors['Flow'] = self.flow_loss
        ret_errors['Warp'] = self.warp_loss
        ret_errors['dmap_warp'] = self.dmap_warp_loss
        return ret_errors

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)

