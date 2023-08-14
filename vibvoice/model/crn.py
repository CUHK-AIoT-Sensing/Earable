'''
based on CRUSE
'''
import torch
import torch.nn as nn
from .vibvoice import synthetic
from .base_model import Dual_RNN_Block, CausalConvBlock, CausalTransConvBlock
from .skip_rnn import Skip_Dual_RNN_Blockclass  
import numpy as np

class CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, add=True ):
        super(CRN, self).__init__()
        self.add = add
        channel_list = [16, 32, 64, 128, 256 ]
        self.vib_conv1 = CausalConvBlock(1, 16)
        self.vib_conv2 = CausalConvBlock(16, 32)
        self.vib_transconv1 = CausalTransConvBlock(32, 16, output_padding=(1, 0))
        self.vib_transconv2 = CausalTransConvBlock(16, 1, is_last=True)

        # Encoder
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(2, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)

        # LSTM
        # self.lstm_layer = nn.LSTM(input_size=256*9, hidden_size=256*9, num_layers=2, batch_first=True)
        self.lstm_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], 'GRU', bidirectional=True  )
        # self.lstm_layer = Skip_Dual_RNN_Blockclass(256, 256, 'GRU')

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
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, 1, is_last=True))
            elif i == 1:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def acc_enhancement(self, acc):
        e_1 = self.vib_conv1(acc)
        e_2 = self.vib_conv2(e_1)
        d_1 = self.vib_transconv1(e_2)
        d_2 = self.vib_transconv2(d_1)
        acc = d_2 * acc
        return acc      

    def forward(self, x, acc):

        acc = self.acc_enhancement(acc)
        pad_acc = torch.nn.functional.pad(acc, (0, 0, 0, x.shape[-2] - acc.shape[-2]))

        Res = []
        d = torch.cat((x, pad_acc), 1)
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)

        # batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape
        # # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        # lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        # lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        # lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]

        d = self.lstm_layer(d)

        if self.add:
            for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
                d = layer(d + skip(Res.pop()))
        else:
            for layer in self.trans_conv_blocks:
                d = layer(torch.cat((d, Res.pop()), 1))

        return d * x, acc

