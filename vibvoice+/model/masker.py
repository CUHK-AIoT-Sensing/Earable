'''
This script based on 
1. "DPCRN: Dual-path convolution recurrent network for single channel speech enhancement"
2. "Fusing Bone-Conduction and Air-Conduction Sensors for Complex-Domain Speech Enhancement"
'''
import torch
import torch.nn as nn
from .base_model import Dual_RNN_Block, CausalConvBlock, CausalTransConvBlock
def masker(model_arch='VibVoice_Lite'):
    if model_arch == 'Baseline': # single-modality
        return Baseline()
    elif model_arch == 'VibVoice':
        return VibVoice() 
    elif model_arch == 'VibVoice_Lite':
        return VibVoice_Lite()
    elif model_arch == 'VibVoice_Early':
        return VibVoice_Early()
    else:
        raise NotImplementedError
class Baseline(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self, channel_list = [16, 32, 64, 128, 256], pad_num=1, last_channel=1, activation=nn.Sigmoid()):
        super(Baseline, self).__init__()
        self.channel_list = channel_list
        self.init_channel = 1

        # Encoder
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(self.init_channel, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)

        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)

        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.skip_convs = nn.ModuleList(layers)
        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            if i == 0:
                layers.append(CausalTransConvBlock(channel_list[i], last_channel, activation=activation))
            elif i == pad_num:
                layers.append(CausalTransConvBlock(channel_list[i], channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i], channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def forward(self, x, acc):
        Res = []
        d = x
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)
        d = self.rnn_layer(d)
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
            d = layer(d + skip(Res.pop()))
        return d * x


    def forward_causal(self, x, acc):
        Res = []
        d = x
        for layer in self.conv_blocks:
            d = layer.forward_causal(d)
            Res.append(d)
        d = self.rnn_layer.forward_causal(d)
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
            d = layer.forward_causal(d + skip(Res.pop()))
        return d * x
class VibVoice(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self, channel_list = [16, 32, 64, 128, 256], pad_num=1, last_channel=1, activation=nn.Sigmoid()):
        super(VibVoice, self).__init__()
        self.channel_list = channel_list
        self.init_channel = 1
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

        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.skip_convs = nn.ModuleList(layers)
       
        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            if i == 0:
                layers.append(CausalTransConvBlock(channel_list[i], last_channel, activation=activation))
            elif i == pad_num:
                layers.append(CausalTransConvBlock(channel_list[i], channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i], channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def forward(self, x, acc):
        Res = []
        d = x
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)
        for layer in self.acc_conv_blocks:
            acc = layer(acc)
        d = d + self.map(acc)

        d = self.rnn_layer(d)
        
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
            d = layer(d + skip(Res.pop()))
        return d * x


    def forward_causal(self, x, acc):
        Res = []
        d = x
        for layer in self.conv_blocks:
            d = layer.forward_causal(d)
            Res.append(d)
        for layer in self.acc_conv_blocks:
            acc = layer.forward_causal(acc)
        d = d + self.map(acc)

        d = self.rnn_layer.forward_causal(d)
        
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
            d = layer.forward_causal(d + skip(Res.pop()))
        return d * x
class VibVoice_Early(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self, channel_list = [16, 32, 64, 128, 256], pad_num=1, last_channel=1, activation=nn.Sigmoid()):
        super(VibVoice_Early, self).__init__()
        self.channel_list = channel_list
        self.init_channel = 2
        # Encoder
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(self.init_channel, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)

        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)

        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.skip_convs = nn.ModuleList(layers)
       
        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            if i == 0:
                layers.append(CausalTransConvBlock(channel_list[i], last_channel, activation=activation))
            elif i == pad_num:
                layers.append(CausalTransConvBlock(channel_list[i], channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i], channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def forward(self, x, acc):
        Res = []
        d = torch.cat((x, acc), 1)
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)

        d = self.rnn_layer(d)
        
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
            d = layer(d + skip(Res.pop()))
        return d * x
    def forward_causal(self, x, acc):
        Res = []
        d = torch.cat((x, acc), 1)
        for layer in self.conv_blocks:
            d = layer.forward_causal(d)
            Res.append(d)

        d = self.rnn_layer.forward_causal(d)
        
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
            d = layer.forward_causal(d + skip(Res.pop()))
        return d * x
class VibVoice_Lite(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self, channel_list = [16, 32, 64, 128], pad_num=1, last_channel=1, activation=nn.Sigmoid()):
        super(VibVoice_Lite, self).__init__()
        self.channel_list = channel_list
        self.init_channel = 2
        # Encoder
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(self.init_channel, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)

        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)

        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.skip_convs = nn.ModuleList(layers)
       
        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            if i == 0:
                layers.append(CausalTransConvBlock(channel_list[i], last_channel, activation=activation))
            elif i == pad_num:
                layers.append(CausalTransConvBlock(channel_list[i], channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i], channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def forward(self, x, acc):
        Res = []
        d = torch.cat((x, acc), 1)
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)

        d = self.rnn_layer(d)
        
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
            d = layer(d + skip(Res.pop()))
        return d * x


    def forward_causal(self, x, acc):
        Res = []
        d = torch.cat((x, acc), 1)
        for layer in self.conv_blocks:
            d = layer.forward_causal(d)
            Res.append(d)

        d = self.rnn_layer.forward_causal(d)
        
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
            d = layer.forward_causal(d + skip(Res.pop()))
        return d * x
    
class DPCRN_masker(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self, channel_list = [16, 32, 64, 128, 256], single_modality=False, early_fusion=False , add=True, pad_num=1, last_channel=1, activation=nn.Sigmoid()):
        super(DPCRN_masker, self).__init__()
        self.single_modality = single_modality
        self.add = add
        self.early_fusion = early_fusion
        self.channel_list = channel_list
        if self.single_modality:
            assert early_fusion == True; "if single_modality, early_fusion must be True"

        if self.early_fusion:
            if self.single_modality:
                self.init_channel = 1
            else:
                self.init_channel = 2
        else:
            self.init_channel = 1
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
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, last_channel, activation=activation))
            elif i == pad_num:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def forward(self, x, acc):
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
