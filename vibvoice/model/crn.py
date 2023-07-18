import torch
import torch.nn as nn
from .vibvoice import synthetic
from .tcnn import TCNN_Block
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


class CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN, self).__init__()
        self.vib_conv1 = CausalConvBlock(1, 16)
        self.vib_conv2 = CausalConvBlock(16, 32)
        self.vib_transconv1 = CausalTransConvBlock(32, 16, output_padding=(1, 0))
        self.vib_transconv2 = CausalTransConvBlock(16, 1, is_last=True)
        # Encoder
        self.conv_block_1 = CausalConvBlock(3, 16)
        self.conv_block_2 = CausalConvBlock(16, 32)
        self.conv_block_3 = CausalConvBlock(32, 64)
        self.conv_block_4 = CausalConvBlock(64, 128)
        self.conv_block_5 = CausalConvBlock(128, 256)

        # LSTM
        # self.lstm_layer = nn.LSTM(input_size=256*9, hidden_size=256*9, num_layers=2, batch_first=True)
        # TCM
        # self.TCNN_Block_1 = TCNN_Block(in_channels=256*9, out_channels=1024, kernel_size=3, init_dilation=2, num_layers=3)
        # self.TCNN_Block_2 = TCNN_Block(in_channels=256*9, out_channels=1024, kernel_size=3, init_dilation=2, num_layers=3)
        # self.TCNN_Block_3 = TCNN_Block(in_channels=256*9, out_channels=1024, kernel_size=3, init_dilation=2, num_layers=3)

        self.tran_conv_block_1 = CausalTransConvBlock(0 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

    def acc_enhancement(self, acc):
        e_1 = self.vib_conv1(acc)
        e_2 = self.vib_conv2(e_1)
        d_1 = self.vib_transconv1(e_2)
        d_2 = self.vib_transconv2(d_1)
        acc = d_2 * acc
        return acc      

    def forward(self, x, acc):
        x = torch.unsqueeze(x, 1)
        acc = torch.unsqueeze(acc, 1)
        # self.lstm_layer.flatten_parameters()

        acc = self.acc_enhancement(acc)
        pad_acc = torch.nn.functional.pad(acc, (0, 0, 0, 321 - 33))
        e_1 = self.conv_block_1(torch.cat((x, pad_acc, (x+pad_acc)/2), 1))
        e_2 = self.conv_block_2(e_1)
        e_3 = self.conv_block_3(e_2)
        e_4 = self.conv_block_4(e_3)
        e_5 = self.conv_block_5(e_4)  # [2, 256, 4, 200]

        # batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape
        # # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        # lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        # lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        # lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]

        # batch_size, C, frame_len, frame_num = e_5.shape
        # reshape_1 = e_5.reshape(batch_size, C * frame_len, frame_num)

        # TCNN_Block_1 = self.TCNN_Block_1(reshape_1)
        # TCNN_Block_2 = self.TCNN_Block_2(TCNN_Block_1)
        # TCNN_Block_3 = self.TCNN_Block_3(TCNN_Block_2)

        # reshape_2 = TCNN_Block_3.reshape(batch_size, C, frame_len, frame_num)

        # d_1 = self.tran_conv_block_1(torch.cat((lstm_out, e_5), 1))
        d_1 = self.tran_conv_block_1(e_5)
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1))
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1))
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1))
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1))

        return d_5 * x, acc


    
if __name__ == '__main__':
    model = CRN()
    
    def constructor(resolution):
        audio = torch.rand(1, 321, 150)
        acc = torch.rand(1, 33, 150)
        return dict(x=audio, acc=acc)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, input_res=(1, 321, 150), as_strings=True, input_constructor=constructor,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))