'''
This script contains audio-only baseline
'''
import torch
import torch.nn as nn
from base_model import Dual_RNN_Block, CausalConvBlock, CausalTransConvBlock

class DPCRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self, channel_list = [16, 32, 64, 128, 256], pad_num=1, last_channel=1, last_act=True):
        super(DPCRN, self).__init__()
        self.channel_list = channel_list

        # Encoder
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(1, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)

        # RNN, try to keep bidirectional=False
        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)
        num_c = 1
        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.skip_convs = nn.ModuleList(layers)
        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            if i == 0:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, last_channel, is_last=last_act))
            elif i == pad_num:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1], output_padding=(1, 0)))
            else:
                layers.append(CausalTransConvBlock(channel_list[i]*num_c, channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def forward(self, x):
        Res = []
        d = x
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)
        d = self.rnn_layer(d)
        for layer, skip in zip(self.trans_conv_blocks, self.skip_convs):
                d = layer(d + skip(Res.pop()))
        return d
    
if __name__ == "__main__":
    model = DPCRN()

    ckpt = torch.load('checkpoints/20230907-142507/best.pth')
    model.load_state_dict(ckpt)
    model.eval()
    dummy = torch.randn(1, 1, 321, 251)
    output = model(dummy)
    print(output.shape)

