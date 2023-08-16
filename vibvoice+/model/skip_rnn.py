'''
this script implements the skip rnn model
we support 2 ways of skipping
1. skip the whole frame based on vad output
2. skip partial of frame based on predicted confidence

(TODO) we plan to support two different models
1. Conv + Dual-RNN
2. Conformer (causal)
They are much efficient and effective than LSTM baseline
'''
import torch
import torch.nn as nn
from .base_model import Dual_RNN_Block

class Confidence_Predictor(nn.Module):
    '''
    implement based on dynamic slimmable network, it is causal
    '''
    def __init__(self):
        super(Confidence_Predictor, self).__init__()
        
        self.map = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        batch_size, channels, freq, time = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().reshape(-1, channels)
        x = self.map(x).reshape(batch_size, time, freq).permute(0, 2, 1).contiguous()
        x = torch.nn.functional.gumbel_softmax(x, hard=True, dim=1)
        return x

class Skip_Dual_RNN_Blockclass(Dual_RNN_Block):
    def __init__(self, out_channels, hidden_channels, rnn_type='LSTM', norm='ln', dropout=0, bidirectional=False):
        super(Skip_Dual_RNN_Blockclass, self).__init__(out_channels,
                 hidden_channels, rnn_type=rnn_type, norm=norm, dropout=dropout, bidirectional=bidirectional)
        self.confidence_predictor = Confidence_Predictor()
        self.tri = torch.tril(torch.ones(9, 9))
    def intra_forward(self, x):
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
        return intra_rnn
    def inter_forward(self, intra_rnn):
        B, N, K, S = intra_rnn.shape
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
    def forward(self, x):
        '''
           x: [B, N, K, S] Batch, feature, frequency, time
           out: [B, N, K, S]
        we select first k frequency bins based on self.skip
        for dual-rnn, 
        1) intra_forward: frequency is reshaped to time dimension, selecting the first k frequency bins will hurt the remaining other frequency bins.
        we may give up enhancement on the remaining frequency bins (training-free, but intra-rnn should be uni-directional, bidirection = False)
        or we may use a mask to mask out the remaining frequency bins (training-required)
        2) inter_forward: frequency is reshaped to batch, so selecting the first k frequency bins will not hurt other frequency bins 
        '''
        confidence = self.confidence_predictor(x)
        confidence = torch.matmul(confidence.permute(0, 2, 1), self.tri).permute(0, 2, 1)
        x *= confidence
        if self.training:
            intra_rnn = self.intra_forward(x)
            out = self.inter_forward(intra_rnn)
        else:
            confidence = confidence.argmax(dim=1)
            # needs to be implemented with causal inference
            intra_rnn = self.intra_forward(x)
            out = self.inter_forward(intra_rnn)
        return out