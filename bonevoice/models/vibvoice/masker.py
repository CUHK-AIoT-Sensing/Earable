'''
This script based on 
1. "DPCRN: Dual-path convolution recurrent network for single channel speech enhancement"
2. "Fusing Bone-Conduction and Air-Conduction Sensors for Complex-Domain Speech Enhancement"
'''
import torch
import torch.nn as nn
from .base_model import Dual_RNN_Block, CausalConvBlock, CausalTransConvBlock
from asteroid_filterbanks import make_enc_dec
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

        self.kernel_size = 640
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

    def encode(self, x):
        stft = torch.stft(x, n_fft=self.kernel_size, hop_length=self.kernel_size//2, return_complex=True, 
                          window=torch.hann_window(self.kernel_size).to(x.device))
        stft_mag, stft_phase = stft.abs(), stft.angle()
        return stft_mag.unsqueeze(1), stft_phase.unsqueeze(1)
    
    def forward(self, x):
        x, x_phase = self.encode(x)
        Res = []
        d = x
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)
        d = self.rnn_layer(d)
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
            d = layer(d + skip(Res.pop()))
        x = d * x
        # return the complex by mag and phase
        x = torch.complex(x * torch.cos(x_phase), x * torch.sin(x_phase)).squeeze(1)
        x = torch.istft(x, n_fft=self.kernel_size, hop_length=self.kernel_size//2, window=torch.hann_window(self.kernel_size).to(x.device))
        return x

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
    def __init__(self, channel_list = [16, 32, 64, 128, 256], last_channel=1, activation=nn.Sigmoid()):
        super(VibVoice, self).__init__()
        
        self.kernel_size = 640
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
        encode_channel = [self.init_channel] + channel_list
        for i in range(len(encode_channel)-1):
            layers.append(CausalConvBlock(encode_channel[i], encode_channel[i+1]))

        self.conv_blocks = nn.ModuleList(layers)

        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)

        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.skip_convs = nn.ModuleList(layers)
       
        layers = []
        decode_channel = channel_list[::-1] + [last_channel]
        for i in range(len(decode_channel)-1):
            if i in [3]:
                layers.append(CausalTransConvBlock(decode_channel[i], decode_channel[i+1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(decode_channel[i], decode_channel[i+1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

        self.activation = activation

    def encode(self, x):
        stft = torch.stft(x, n_fft=self.kernel_size, hop_length=self.kernel_size//2, return_complex=True, 
                          window=torch.hann_window(self.kernel_size).to(x.device))
        stft_mag, stft_phase = stft.abs(), stft.angle()
        return stft_mag.unsqueeze(1), stft_phase.unsqueeze(1)
    
    def forward(self, x, acc):
        x, x_phase = self.encode(x); acc, _ = self.encode(acc)
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
        d = self.activation(d)
        x = d * x
        # return the complex by mag and phase
        x = torch.complex(x * torch.cos(x_phase), x * torch.sin(x_phase)).squeeze(1)
        x = torch.istft(x, n_fft=self.kernel_size, hop_length=self.kernel_size//2, window=torch.hann_window(self.kernel_size).to(x.device))
        return x

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
