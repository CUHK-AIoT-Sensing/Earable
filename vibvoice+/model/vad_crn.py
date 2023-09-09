'''
similar strucutre as 
1. "In-Ear-Voice: Towards Milli-Watt Audio Enhancement With Bone-Conduction Microphones for In-Ear Sensing Platforms, IoTDI'23"
2. "On training targets for noise-robust voice activity detection"
'''
import torch
import torch.nn as nn
from .base_model import CausalConvBlock, CausalTransConvBlock



class VAD_CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, rnn_type='GRU'):
        super(VAD_CRN, self).__init__()
        self.vib_conv1 = CausalConvBlock(1, 16)
        self.vib_conv2 = CausalConvBlock(16, 32)
        self.vib_conv3 = CausalConvBlock(32, 64)
        self.vib_conv4 = CausalConvBlock(64, 128)

        self.rnn_layer = getattr(nn, rnn_type)(128*3, 128*3, 2, batch_first=True)
        self.fc1 = nn.Linear(128*3, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        acc = self.vib_conv1(x)
        acc = self.vib_conv2(acc)
        acc = self.vib_conv3(acc)
        acc = self.vib_conv4(acc)

        batch_size, n_channels, n_f_bins, n_frame_size = acc.shape
        acc = acc.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        acc, _ = self.rnn_layer(acc)
        vad = nn.functional.relu(self.fc1(acc))
        vad = torch.sigmoid(self.fc2(vad)).reshape(-1, 1)
        return vad
    


class VAD_decoder(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, channel=128*3, rnn_type='GRU'):
        super(VAD_decoder, self).__init__()
        self.channel = channel
        self.rnn_layer = getattr(nn, rnn_type)(channel, channel, 1, batch_first=True)
        self.fc1 = nn.Linear(channel, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size, n_channels, n_f_bins, n_frame_size = x.shape
        x = x.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        x, _ = self.rnn_layer(x[:, :, :self.channel])
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x