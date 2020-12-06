import os
import torch
import torch.nn as nn

from .networks import get_grid

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        nb = opt.batchSize
        size = opt.fineSize
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.input_prev_I_set = self.Tensor(nb, opt.P_input_nc * (opt.n_frames_G - 1), size, size)
        self.input_prev_D_set = self.Tensor(nb, opt.BP_input_nc * (opt.n_frames_G - 1), size, size)
        self.input_last_I_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_last_D_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_curr_I_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_curr_D_set = self.Tensor(nb, opt.BP_input_nc, size, size)

    def set_input(self, input):
        input_prev_I_set = input['I'][:,:(self.tG-1)*self.P_input_nc]
        input_prev_D_set = input['D'][:,:(self.tG-1)*self.BP_input_nc]
        input_last_I_set = input['I'][:,(self.tG-2)*self.P_input_nc:(self.tG-1)*self.P_input_nc]
        input_last_D_set = input['D'][:,(self.tG-2)*self.BP_input_nc:(self.tG-1)*self.BP_input_nc]
        input_curr_I_set = input['I'][:,(self.tG-1)*self.P_input_nc:]
        input_curr_D_set = input['D'][:,(self.tG-1)*self.BP_input_nc:]
        self.input_prev_I_set.resize_(input_prev_I_set.size()).copy_(input_prev_I_set)
        self.input_prev_D_set.resize_(input_prev_D_set.size()).copy_(input_prev_D_set)
        self.input_last_I_set.resize_(input_last_I_set.size()).copy_(input_last_I_set)
        self.input_last_D_set.resize_(input_last_D_set.size()).copy_(input_last_D_set)
        self.input_curr_I_set.resize_(input_curr_I_set.size()).copy_(input_curr_I_set)
        self.input_curr_D_set.resize_(input_curr_D_set.size()).copy_(input_curr_D_set)

        self.image_paths = 'Gen_'+input['I_path'][0]

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def grid_sample(self, input1, input2):
        return torch.nn.functional.grid_sample(input1, input2, mode='bilinear', padding_mode='border')

    def resample(self, image, flow):
        b, c, h, w = image.size()
        if not hasattr(self, 'grid') or self.grid.size() != flow.size():
            self.grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
        final_grid = (self.grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
        output = self.grid_sample(image, final_grid)
        return output
