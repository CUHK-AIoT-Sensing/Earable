'''
similar strucutre as "In-Ear-Voice: Towards Milli-Watt Audio Enhancement With Bone-Conduction Microphones for In-Ear Sensing Platforms, IoTDI'23"
'''
import torch
import torch.nn as nn
from .vibvoice import synthetic
import numpy as np

class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class VAD(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, rnn_type='GRU'):
        super(VAD, self).__init__()
        self.vib_conv1 = CausalConvBlock(1, 16)
        self.vib_conv2 = CausalConvBlock(16, 32)
        self.rnn_layer = getattr(nn, rnn_type)(32*7, 32, 2, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, acc):
        acc = self.vib_conv1(acc)
        acc = self.vib_conv2(acc)
        batch_size, n_channels, n_f_bins, n_frame_size = acc.shape
        acc = acc.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        acc, _ = self.rnn_layer(acc)
        acc = nn.functional.relu(self.fc1(acc))
        acc = torch.sigmoid(self.fc2(acc))
        return acc