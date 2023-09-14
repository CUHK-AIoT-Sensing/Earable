'''
This script based on 
1. "DPCRN: Dual-path convolution recurrent network for single channel speech enhancement"
2. "Fusing Bone-Conduction and Air-Conduction Sensors for Complex-Domain Speech Enhancement"
'''
import torch
import torch.nn as nn
from .base_model import Dual_RNN_Block, CausalConvBlock, CausalTransConvBlock
class SUPER(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, channel_list = [16, 32, 64, 128, 256], add=False, pad_num=1, last_channel=1, last_act=True):
        super(SUPER, self).__init__()
        self.add = add
        self.channel_list = channel_list

        init_channel = 1
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(init_channel, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)

        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)
     
        num_c = 2
        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            if i == 0:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, last_channel, is_last=last_act))
            elif i == pad_num:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def forward(self, x, acc):
        Res = []
        d = acc
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)
        d = self.rnn_layer(d)
        for layer in self.trans_conv_blocks:
            d = layer(torch.cat((d, Res.pop()), 1))
        return d

