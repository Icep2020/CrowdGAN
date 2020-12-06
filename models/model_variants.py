import torch.nn as nn
import torch
import functools
import torch.nn.functional as F
from torch.autograd import Variable

class FlowEstModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=[6],
                 gpu_ids=[], padding_type='reflect', n_downsampling=2, fusion_stage=False):
        assert(type(input_nc)==list and type(n_blocks)==list)
        super(FlowEstModel, self).__init__()
        self.input_nc = input_nc[0]
        self.output_nc = output_nc
        self.n_blocks = n_blocks[0]
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.fusion = fusion_stage
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down_sample
        model_stream_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]


        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model_stream_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]

        # att_block in place of res_block
        mult = 2**n_downsampling

        resBlock = nn.ModuleList()
        for i in range(self.n_blocks):
            resBlock.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias))

        # up_sample
        model_stream_up = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_stream_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]

        model_stream_up += [nn.ReflectionPad2d(3)]
        model_stream_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        # self.model = nn.Sequential(*model)
        self.stream_down = nn.Sequential(*model_stream_down)
        # self.att = nn.Sequential(*attBlock)
        self.att = resBlock
        self.stream_up = nn.Sequential(*model_stream_up)

    def forward(self, input): # x from stream 1 and stream 2
        # down_sample
        x = self.stream_down(input)

        # att_block
        for model in self.att:
            x = model(x)

        if self.fusion:
            feature = x

        # up_sample
        x = self.stream_up(x)

        return {'out':x, 'fea':feature} if self.fusion else x

class FlowEstNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer, blocks_G, gpu_ids, padding_type='reflect', n_downsampling=2, fusion_stage=False):
        super(FlowEstNet, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 1, 'The AttModule take input_nc in format of list only!!'
        assert type(blocks_G) == list and len(blocks_G) == 1, 'The AttModule take blocks_G in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = FlowEstModel(input_nc, output_nc, ngf, norm_layer, blocks_G, gpu_ids, padding_type, n_downsampling=n_downsampling, fusion_stage=fusion_stage)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class SATBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_bias, cated_stream2=False, use_dropout=False):
        super(SATBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_bias, cal_att=False, use_dropout=use_dropout)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_bias, cal_att=True, cated_stream2=cated_stream2, use_dropout=use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias, cated_stream2=False, cal_att=False, use_dropout=False):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cated_stream2:
            conv_block += [nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim*2),
                       nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        if cal_att:
            if cated_stream2:
                conv_block += [nn.Conv2d(dim*2, dim, kernel_size=3, padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        att = F.sigmoid(x2_out)

        x1_out = x1_out * att
        out = x1 + x1_out # residual connection

        # stream2 receive feedback from stream1
        return out, x2_out, x1_out

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = nn.ModuleList()
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = cell_list

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class TransferModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=[4,4],
                 gpu_ids=[], padding_type='reflect', n_downsampling=2, use_dropout=False, input_size=(256, 320), fusion_stage=False):
        assert(type(input_nc)==list and type(n_blocks)==list)
        super(TransferModel, self).__init__()
        self.image_nc = input_nc[0]
        self.dmap_nc = input_nc[1]
        self.prev_nc = input_nc[0]
        self.n_dmap_prev = n_blocks[0]
        self.n_dmap_post = n_blocks[1]
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.fusion = fusion_stage
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down sample
        dmap_branch_stream1_down = [nn.ReflectionPad2d(3),
                                    nn.Conv2d(self.image_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                    norm_layer(ngf),
                                    nn.ReLU(True)]

        dmap_branch_stream2_down = [nn.ReflectionPad2d(3),
                                    nn.Conv2d(self.dmap_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                    norm_layer(ngf),
                                    nn.ReLU(True)]

        dmap_branch_stream3_down = [nn.ReflectionPad2d(3),
                                    nn.Conv2d(self.prev_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                    norm_layer(ngf),
                                    nn.ReLU(True)]


        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            dmap_branch_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                                   stride=2, padding=1, bias=use_bias),
                                         norm_layer(ngf * mult * 2),
                                         nn.ReLU(True)]

            dmap_branch_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                                   stride=2, padding=1, bias=use_bias),
                                         norm_layer(ngf * mult * 2),
                                         nn.ReLU(True)]

            dmap_branch_stream3_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                                   stride=2, padding=1, bias=use_bias),
                                         norm_layer(ngf * mult * 2),
                                         nn.ReLU(True)]

        mult = 2 ** n_downsampling

        self.temporal_path = ConvLSTM(input_size=(int(input_size[0] / mult), int(input_size[1] / mult)),
                 input_dim=ngf*mult,
                 hidden_dim=[ngf, ngf, ngf*mult],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=use_bias,
                 return_all_layers=False)

        # SATBlock_prev
        cated_stream2_prev = [False for i in range(self.n_dmap_prev)]
        SATBlock_prev = nn.ModuleList()
        for i in range(self.n_dmap_prev):
            SATBlock_prev.append(SATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, cated_stream2=cated_stream2_prev[i], use_dropout=use_dropout))

        # ResBlock_prev
        #ResBlock_prev = nn.ModuleList()
        #for i in range(self.n_dmap_prev):
        #    ResBlock_prev.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias))


        inter_conv_block = []
        p = 0
        if padding_type == 'reflect':
            inter_conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            inter_conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        inter_conv_block += [nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, padding=p, bias=use_bias),
                             norm_layer(ngf * mult)]

        # SATBlock_post
        cated_stream2_post = [False for i in range(self.n_dmap_post)]
        SATBlock_post = nn.ModuleList()
        for i in range(self.n_dmap_post):
            SATBlock_post.append(SATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, cated_stream2=cated_stream2_post[i], use_dropout=use_dropout))

        # up sample
        dmap_branch_stream1_up = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            dmap_branch_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(int(ngf * mult / 2)),
                                       nn.ReLU(True)]

        dmap_branch_stream1_up += [nn.ReflectionPad2d(3),
                                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

        self.dmap_branch_stream1_down = nn.Sequential(*dmap_branch_stream1_down)
        self.dmap_branch_stream2_down = nn.Sequential(*dmap_branch_stream2_down)
        self.dmap_branch_stream3_down = nn.Sequential(*dmap_branch_stream3_down)
        self.dmap_branch_prev = SATBlock_prev
        self.inter_conv = nn.Sequential(*inter_conv_block)
        self.dmap_branch_post = SATBlock_post
        self.dmap_branch_stream1_up = nn.Sequential(*dmap_branch_stream1_up)

    def forward(self, input):
        x1, x2, t = input
        b,_,h,w = t.size()
        t = t.view(-1, 3, h, w)
        # down sample
        x1 = self.dmap_branch_stream1_down(x1)
        x2 = self.dmap_branch_stream2_down(x2)
        t = self.dmap_branch_stream3_down(t)
        # dmap branch
        for model in self.dmap_branch_prev:
            x1, x2, _ = model(x1, x2)

        # temporal path
        _,c,h,w = t.size()
        t = t.view(b, -1, c, h, w)
        t,_ = self.temporal_path(t)
        t = t[0][:,-1]

        # inter
        x2 = self.inter_conv(torch.cat([x2, t],dim=1))

        # dmap branch
        for model in self.dmap_branch_post:
            x1, x2, _ = model(x1, x2)

        if self.fusion:
            feature = x1

        # up sample
        x1 = self.dmap_branch_stream1_up(x1)

        return {'out':x1, 'fea':feature} if self.fusion else x1

class TransferNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer, blocks_G, gpu_ids, padding_type='reflect', n_downsampling=2, use_dropout=False, input_size=(256, 320), fusion_stage=False):
        super(TransferNet, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 3, 'The AttModule take input_nc in format of list only!!'
        assert type(blocks_G) == list and len(blocks_G) == 2, 'The AttModule take blocks_G in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = TransferModel(input_nc, output_nc, ngf, norm_layer, blocks_G, gpu_ids, padding_type, n_downsampling=n_downsampling, use_dropout=use_dropout, input_size=input_size, fusion_stage=fusion_stage)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class FusionModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=[6],
                 gpu_ids=[], padding_type='reflect', n_downsampling=2):
        assert(type(input_nc)==list and type(n_blocks)==list)
        super(FusionModel, self).__init__()
        self.input_nc = input_nc[0]
        self.output_nc = output_nc
        self.n_blocks = n_blocks[0]
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        mult = 2**n_downsampling

        resBlock = nn.ModuleList()
        for i in range(self.n_blocks):
            resBlock.append(ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias))

        # up_sample
        model_stream_up = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_stream_up += [nn.ConvTranspose2d(ngf * mult * 2 , ngf * mult,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                            norm_layer(ngf * mult),
                            nn.ReLU(True)]

        model_stream_up += [nn.ReflectionPad2d(3)]
        model_stream_up += [nn.Conv2d(ngf * 2, output_nc, kernel_size=7, padding=0)]
        model_stream_up += [nn.Sigmoid()]

        self.att = resBlock
        self.stream_up = nn.Sequential(*model_stream_up)

    def forward(self, input):
        # down_sample
        x, f = input
        x = torch.cat((x, f), 1)

        # att_block
        for model in self.att:
            x = model(x)

        # up_sample
        x = self.stream_up(x)

        return x

class FusionNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer, blocks_G, gpu_ids, padding_type='reflect', n_downsampling=2):
        super(FusionNet, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 1, 'The AttModule take input_nc in format of list only!!'
        assert type(blocks_G) == list and len(blocks_G) == 1, 'The AttModule take blocks_G in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = FusionModel(input_nc, output_nc, ngf, norm_layer, blocks_G, gpu_ids, padding_type, n_downsampling=n_downsampling)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
