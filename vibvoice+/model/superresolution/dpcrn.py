'''
This script based on 
1. "DPCRN: Dual-path convolution recurrent network for single channel speech enhancement"
2. "Fusing Bone-Conduction and Air-Conduction Sensors for Complex-Domain Speech Enhancement"
'''
import torch
import torch.nn as nn
from ..base_model import Dual_RNN_Block, CausalConvBlock, CausalTransConvBlock
from torch.cuda.amp import autocast 
from .spec import spectro, ispectro
from .stft_loss import MultiResolutionSTFTLoss
import numpy as np

class DPCRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self, channel_list = [16, 32, 64, 128, 256], single_modality=True, real_imag=False, early_fusion=True, 
                 add=True, pad_num=1, last_channel=1):
        super(DPCRN, self).__init__()
        self.single_modality = single_modality
        if self.single_modality:
            assert early_fusion == True; "if single_modality, early_fusion must be True"
        self.add = add
        self.early_fusion = early_fusion
        self.channel_list = channel_list
        self.real_imag = real_imag

        self.init_channel = 0
        if self.real_imag:
            self.init_channel += 2
        else:
            self.init_channel += 1
        if self.single_modality:
            self.init_channel += 0
        elif self.early_fusion:
            self.init_channel += 1
        else:
            self.init_channel += 0

        if not self.early_fusion:
            layers = []
            for i in range(len(channel_list)):
                if i == 0:
                    layers.append(CausalConvBlock(1, channel_list[i]))
                else:
                    layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
            self.acc_conv_blocks = nn.ModuleList(layers)
            self.map = nn.Conv2d(channel_list[-1], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Encoder
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(self.init_channel, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)

        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)

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
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, last_channel, activation=nn.Identity()))
            elif i == pad_num:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)
        self.mrstftloss = MultiResolutionSTFTLoss()

        self.scale = 4
        self.nfft = 640
        self.hop_length = int(320 // self.scale)  # this is for the input signal
        self.win_length = int(640 // self.scale)  # this is for the input signal
    def loss(self, lr, hr):
        return self.mrstftloss(lr, hr)
    def _spec(self, x, scale=False):
        if np.mod(x.shape[-1], self.hop_length):
            x = torch.nn.functional.pad(x, (0, self.hop_length - np.mod(x.shape[-1], self.hop_length)))
        hl = self.hop_length
        nfft = self.nfft
        win_length = self.win_length

        if scale:
            hl = int(hl * self.scale)
            win_length = int(win_length * self.scale)

        z = spectro(x, nfft, hl, win_length=win_length)[..., :-1, :]
        return z

    def _ispec(self, z):
        hl = int(self.hop_length * self.scale)
        win_length = int(self.win_length * self.scale)
        z = torch.nn.functional.pad(z, (0, 0, 0, 1))
        x = ispectro(z, hl, win_length=win_length)
        return x

    def _move_complex_to_channels_dim(self, z):
        B, C, Fr, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, Fr, T)
        return m

    def _convert_to_complex(self, x):
        """

        :param x: signal of shape [Batch, Channels, 2, Freq, TimeFrames]
        :return: complex signal of shape [Batch, Channels, Freq, TimeFrames]
        """
        out = x.permute(0, 1, 3, 4, 2)
        out = torch.view_as_complex(out.contiguous())
        return out
    def forward(self, x, acc):
        z = self._spec(x)
        x = self._move_complex_to_channels_dim(z)

        # unlike previous Demucs, we always normalize because it is easier. we can use moving normalization instead
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        x = self.forward_legacy(x, acc)
        x = x * std[:, None] + mean[:, None]
        x_spec_complex = self._convert_to_complex(x)
        x = self._ispec(x_spec_complex)
        return x
    def forward_legacy(self, x, acc):
        Res = []
        if self.early_fusion:
            if self.single_modality:
                d = x
            else:
                d = torch.cat((x, acc), 1)
            for layer in self.conv_blocks:
                d = layer(d)
                Res.append(d)
        else:
            d = x
            for layer in self.conv_blocks:
                d = layer(d)
                Res.append(d)
            for layer in self.acc_conv_blocks:
                acc = layer(acc)
            d = d + self.map(acc)

        if dvector is not None:
            dvector = torch.tile(dvector[:, :, None, None], (1, 1, 1, d.shape[-1]))
            d = torch.cat((d, dvector), 2)
            d = self.rnn_layer(d)
            d = d[:, :, :-1, :]
        else:
            d = self.rnn_layer(d)
        if self.add:
            for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
                d = layer(d + skip(Res.pop()))
        else:
            for layer in self.trans_conv_blocks:
                d = layer(torch.cat((d, Res.pop()), 1))
        return d * x

    def forward_causal(self, x, acc):
        Res = []
        if self.early_fusion:
            if self.single_modality:
                d = x
            else:
                d = torch.cat((x, acc), 1)
            for layer in self.conv_blocks:
                d = layer.forward_causal(d)
                Res.append(d)

        else:
            d = x
            for layer in self.conv_blocks:
                d = layer.forward_causal(d)
                Res.append(d)
            for layer in self.acc_conv_blocks:
                acc = layer.forward_causal(acc)
            d = d + self.map(acc)

        d = self.rnn_layer.forward_causal(d)

        if self.add:
            for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
                d = layer.forward_causal(d + skip(Res.pop()))
        else:
            for layer in self.trans_conv_blocks:
                d = layer.forward_causal(torch.cat((d, Res.pop()), 1))
        return d * x
