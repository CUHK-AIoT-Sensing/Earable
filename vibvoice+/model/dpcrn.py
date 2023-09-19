'''
This script based on 
1. "DPCRN: Dual-path convolution recurrent network for single channel speech enhancement"
2. "Fusing Bone-Conduction and Air-Conduction Sensors for Complex-Domain Speech Enhancement"
'''
import torch
import torch.nn as nn
from .base_model import Dual_RNN_Block, CausalConvBlock, CausalTransConvBlock
from torch.cuda.amp import autocast 
class encoder(nn.Module):
    def __init__(self, channel_list = [16, 32, 64, 128, 256], init_channel=1):
        super(encoder, self).__init__()
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(init_channel, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)
    def forward(self, x):
        Res = []
        for layer in self.conv_blocks:
            x = layer(x)
            Res.append(x)
        return x, Res
class decoder(nn.Module):
    def __init__(self, channel_list = [16, 32, 64, 128, 256], add=True, pad_num=1, last_channel=1, last_act=nn.Sigmoid()):
        super(decoder, self).__init__()
        self.add = add
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
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, last_channel, activation=last_act))
            elif i == pad_num:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)
    def forward(self, x, Res):
        if self.add:
            for i, (layer, skip) in enumerate(zip(self.trans_conv_blocks, self.skip_convs)):
                x = layer(x + skip(Res[-(i+1)]))
        else:
            for layer in self.trans_conv_blocks:
                x = layer(torch.cat((x, Res.pop()), 1))
        return x
class DPCRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self, channel_list = [16, 32, 64, 128, 256], single_modality=False, real_imag=False, early_fusion=True, add=True, pad_num=1, last_channel=1, last_act=nn.ReLU()):
        super(DPCRN, self).__init__()
        self.single_modality = single_modality
        if self.single_modality:
            assert early_fusion == True; "if single_modality, early_fusion must be True"
        self.add = add
        self.early_fusion = early_fusion
        self.channel_list = channel_list
        self.real_imag = real_imag

        last_channel =1
        init_channel = 0
        if self.real_imag:
            last_channel = 2
            last_act = nn.Identity()
            self.fc1 = nn.Linear(321, 321)
            self.fc2 = nn.Linear(321, 321)
            init_channel += 2
        else:
            init_channel += 1
        if self.single_modality:
            init_channel += 0
        elif self.early_fusion:
            init_channel += 1
        else:
            init_channel += 0

        if not self.early_fusion:
            self.acc_encoder = encoder(channel_list)
            self.map = nn.Conv2d(channel_list[-1], channel_list[-1], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Encoder
        self.encoder = encoder(channel_list, init_channel)

        # RNN, try to keep bidirectional=False
        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)

        # Decoder
        # self.decoder = decoder(channel_list, add, pad_num, last_channel, last_act)
        self.decoder_mask = decoder(channel_list, True, pad_num, last_channel, nn.Sigmoid())
        self.decoder_map = decoder(channel_list, True, pad_num, last_channel, nn.ReLU())

    # @autocast()
    def forward(self, x, acc):
        Res = []
        if self.early_fusion:
            if self.single_modality:
                d = x
            else:
                d = torch.cat((x, acc), 1)
            d, Res = self.encoder(d)
        else:
            d = x
            d, Res = self.encoder(d)
            acc, _ = self.acc_encoder(acc)
            d = d + self.map(acc)

        d = self.rnn_layer(d)
        mask = self.decoder_mask(d, Res)
        map = self.decoder_map(d, Res)
        return acc * mask + map

