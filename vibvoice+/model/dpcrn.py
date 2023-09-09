'''
This script based on 
1. "DPCRN: Dual-path convolution recurrent network for single channel speech enhancement"
2. "Fusing Bone-Conduction and Air-Conduction Sensors for Complex-Domain Speech Enhancement"
'''
import torch
import torch.nn as nn
from .base_model import Dual_RNN_Block, CausalConvBlock, CausalTransConvBlock
class DPCRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, channel_list = [16, 32, 64, 128, 256], add=True, early_fusion=False, pad_num=1, last_channel=1, last_act=True):
        super(DPCRN, self).__init__()
        self.add = add
        self.early_fusion = early_fusion
        self.channel_list = channel_list

        if self.early_fusion:
            init_channel = 2
        else:
            init_channel = 1
            layers = []
            for i in range(len(channel_list)):
                if i == 0:
                    layers.append(CausalConvBlock(init_channel, channel_list[i]))
                else:
                    layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
            self.acc_conv_blocks = nn.ModuleList(layers)
            self.map = nn.Conv2d(channel_list[-1], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Encoder
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(init_channel, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)

        # RNN, try to keep bidirectional=False
        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)
        # self.rnn_layer = GGRU(hidden_size=512*9, groups=1)
        if self.add:
            num_c = 1
            layers = []
            for i in range(len(channel_list)-1, -1, -1):
                layers.append(nn.Conv2d(channel_list[i], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
            self.skip_convs = nn.ModuleList(layers)
        else:
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
        if self.early_fusion:
            d = torch.cat((x, acc), 1)
            # d = x
            for layer in self.conv_blocks:
                d = layer(d)
                Res.append(d)
            d = self.rnn_layer(d)

        else:
            d = x
            for layer in self.conv_blocks:
                d = layer(d)
                Res.append(d)
            for layer in self.acc_conv_blocks:
                acc = layer(acc)
            # d = torch.cat((d, acc), 1)
            d = d + self.map(acc)
            d = self.rnn_layer(d)

        if self.add:
            for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
                d = layer(d + skip(Res.pop()))
        else:
            for layer in self.trans_conv_blocks:
                d = layer(torch.cat((d, Res.pop()), 1))
        return d

