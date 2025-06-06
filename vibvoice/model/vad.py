'''
similar strucutre as "In-Ear-Voice: Towards Milli-Watt Audio Enhancement With Bone-Conduction Microphones for In-Ear Sensing Platforms, IoTDI'23"
'''
import torch
import torch.nn as nn
from .vibvoice import synthetic
from .base_model import CausalConvBlock, CausalTransConvBlock
import numpy as np



class VAD(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, rnn_type='GRU'):
        super(VAD, self).__init__()
        self.vib_conv1 = CausalConvBlock(1, 16)
        self.vib_conv2 = CausalConvBlock(16, 32)
        self.rnn_layer = getattr(nn, rnn_type)(32*7, 32*7, 2, batch_first=True)
        self.fc1 = nn.Linear(224, 16)
        self.fc2 = nn.Linear(16, 1)
        
        self.trans_convs = nn.Sequential(CausalTransConvBlock(32, 16, output_padding=(1, 0))
                                         ,CausalTransConvBlock(16, 1, is_last=True))

    def forward(self, x):
        acc = self.vib_conv1(x)
        acc = self.vib_conv2(acc)
        batch_size, n_channels, n_f_bins, n_frame_size = acc.shape
        acc = acc.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        acc, _ = self.rnn_layer(acc)
        vad = nn.functional.relu(self.fc1(acc))
        vad = torch.sigmoid(self.fc2(vad))
        acc = acc.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)
        acc = self.trans_convs(acc) * x
        return acc, vad