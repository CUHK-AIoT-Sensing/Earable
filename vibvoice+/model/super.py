import torch
import torch.nn as nn
from .base_model import Dual_RNN_Block, CausalConvBlock, CausalTransConvBlock
from .hifi_gan import Generator
import json
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
class DPCRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, channel_list = [16, 32, 64, 128, 256], add=False, last_channel=1):
        super(DPCRN, self).__init__()
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
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, last_channel, activation=nn.Identity(), output_padding=(1, 0)))
            elif i == 4:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def forward(self, d):
        Res = []
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)
        d = self.rnn_layer(d)
        for layer in self.trans_conv_blocks:
            d = layer(torch.cat((d, Res.pop()), 1))
        return d
class SUPER(nn.Module):
    def __init__(self, mode='dpcrn'):
        super(SUPER, self).__init__()
        self.analysis_module = DPCRN()
        with open('model/config_v2_16k.json') as f:
            data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)
        self.generator = Generator(h)
    def forward(self, x, acc):
        est_mag = self.analysis_module(acc) + acc
        est_wav = self.generator(est_mag.squeeze(1))
        return est_mag, est_wav
