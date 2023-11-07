'''
contains base model:
1. fullsubnet (BaseModel)
2. Dual-RNN
3. causal conv
'''
import torch
import torch.nn as nn
import numpy as np

EPSILON = np.finfo(np.float32).eps

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)
    
class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, in_channels, out_channels,
                 hidden_channels, rnn_type='GRU', norm='cln',
                 dropout=0, bidirectional=False):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            in_channels, hidden_channels//2 if bidirectional else hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            in_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=False)
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Linear(hidden_channels, out_channels)
        self.inter_linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        '''
           x: [B, N, K, S] Batch, feature, frequency, time
           out: [Spks, B, N, K, S]
        '''
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()

        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out
    def forward_causal(self, x, ):
        '''
           x: [B, N, K, S] Batch, feature, frequency, time
           out: [Spks, B, N, K, S]
        '''
    
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()
        # Batch, feature, frequency, time
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        if not hasattr(self, 'cache'):
            self.cache = torch.zeros(x.shape[0] * x.shape[3], x.shape[2], x.shape[1]).to(x.device)
        inter_rnn, h_n = self.inter_rnn(inter_rnn, self.cache)
        self.cache = h_n        
        
        # inter_rnn, h_n = self.inter_rnn(inter_rnn, cache)

        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out

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
    def forward_causal(self, x,):
        if not hasattr(self, 'cache'):
            self.look_before = 1
            self.cache = torch.zeros(x.shape[0], x.shape[1], x.shape[2], self.look_before).to(x.device)
        update_cache = x[..., -self.look_before:]
        x = torch.cat([self.cache, x], dim=-1)
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        x = x[..., self.look_before:]
        self.cache = update_cache
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ELU(), output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation

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
    
    def forward_causal(self, x,):
        if not hasattr(self, 'cache'):
            self.look_before = 1
            self.cache = torch.zeros(x.shape[0], x.shape[1], x.shape[2], self.look_before).to(x.device)
        update_cache = x[..., -self.look_before:]
        x = torch.cat([self.cache, x], dim=-1)
        x = self.conv(x)
        x = x[:, :, :, :-1]
        x = self.norm(x)
        x = self.activation(x)
        x = x[:, :, :, self.look_before:]
        self.cache = update_cache
        return x
    
class DPCRN_basic(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self, channel_list = [16, 32, 64, 128, 256], init_channel=1, pad=[], last_channel=1):
        super(DPCRN_basic, self).__init__()
        # Encoder
        self.pad = pad
        layers = []
        for i in range(len(channel_list)):
            if i == 0:
                layers.append(CausalConvBlock(init_channel, channel_list[i]))
            else:
                layers.append(CausalConvBlock(channel_list[i-1], channel_list[i]))
        self.conv_blocks = nn.ModuleList(layers)

        self.rnn_layer = Dual_RNN_Block(channel_list[-1], channel_list[-1], 'GRU', bidirectional=False)

        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            layers.append(nn.Conv2d(channel_list[i], channel_list[i], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.skip_convs = nn.ModuleList(layers)
        
        layers = []
        for i in range(len(channel_list)-1, -1, -1):
            if i == 0:
                layers.append(CausalTransConvBlock(channel_list[i], last_channel, activation=nn.Identity()))
            else:
                layers.append(CausalTransConvBlock(channel_list[i], channel_list[i-1]))
        self.trans_conv_blocks = nn.ModuleList(layers)

    def forward(self, x):
        Res = []
        d = x
        for layer in self.conv_blocks:
            d = layer(d)
            Res.append(d)

        d = self.rnn_layer(d)

        for i, (layer, skip) in enumerate(zip(self.trans_conv_blocks, self.skip_convs)):
            d = layer(d + skip(Res.pop()))
            if i in self.pad:
                d = torch.nn.functional.pad(d, (0, 0, 0, 1, ), value=0)
        return d
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action="store", type=str, default='Conv', required=False, help='choose the model')
    args = parser.parse_args()
    if args.model == 'Conv':
        # we will test whether causal inference is right
        # test 1: causal convolution
        dummpy_input = torch.randn(1, 1, 321, 251)
        conv = CausalConvBlock(1, 16)
        conv.eval()
        full_output = conv(dummpy_input)
        real_time_output = []
        for i in range(dummpy_input.shape[-1]):
            real_time_input = dummpy_input[:, :, :, i:i+1]
            real_time_output.append(conv.forward_causal(real_time_input,))
        real_time_output = torch.cat(real_time_output, dim=-1)
        error = torch.mean(torch.abs((full_output - real_time_output)/full_output))
        print('Real-time difference ' + args.model, error.item())
    elif args.model == 'TransConv':
        # test 2: causal deconvolution
        dummpy_input = torch.randn(1, 1, 321, 251)
        conv = CausalTransConvBlock(1, 16)
        conv.eval()
        full_output = conv(dummpy_input)    
        real_time_output = []
        for i in range(dummpy_input.shape[-1]):
            real_time_input = dummpy_input[:, :, :, i:i+1]
            real_time_output.append(conv.forward_causal(real_time_input, ))
        real_time_output = torch.cat(real_time_output, dim=-1)
        error = torch.mean(torch.abs((full_output - real_time_output)/full_output))
        print('Real-time difference ' + args.model, error.item())
    elif args.model == 'RNN':
        # test 3: RNN
        dummpy_input = torch.randn(1, 251, 256)
        rnn = nn.RNN(256, 256, 1, batch_first=True, dropout=0, bidirectional=False)
        rnn.eval()
        full_output, h_n = rnn(dummpy_input)
        cache = torch.zeros(1, 1, 256)
        real_time_output = []
        for i in range(dummpy_input.shape[1]):
            real_time_input = dummpy_input[:, i:i+1, :]
            output, h_n = rnn(real_time_input, cache)
            real_time_output.append(output)
            cache = h_n
        real_time_output = torch.cat(real_time_output, dim=1)
        error = torch.mean(torch.abs((full_output - real_time_output)/full_output))
        print('Real-time difference ' + args.model, error.item())
    elif args.model == 'DPRNN':
        # test 4: DPRNN
        dummpy_input = torch.randn(1, 256, 4, 251)
        dprnn = Dual_RNN_Block(256, 256)
        dprnn.eval()
        full_output = dprnn(dummpy_input)
        real_time_output = []
        for i in range(dummpy_input.shape[-1]):
            real_time_input = dummpy_input[:, :, :, i:i+1]
            output = dprnn.forward_causal(real_time_input, )
            real_time_output.append(output)
        real_time_output = torch.cat(real_time_output, dim=-1)
        error = torch.mean(torch.abs((full_output - real_time_output)/full_output))
        print('Real-time difference ' + args.model, error.item())
