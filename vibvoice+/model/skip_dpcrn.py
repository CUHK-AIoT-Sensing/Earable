'''
this script implements the skip rnn model
we support 2 ways of skipping
1. skip time frame based on vad output (time-dimension)
2. skip frequency bins based on predicted confidence (frequency-dimension)
    we select first k frequency bins based on self.skip
    for dual-rnn, 
    1) intra_forward: frequency is reshaped to time dimension, selecting the first k frequency bins will not hurt the remaining other frequency bins (if bidirectional = False), if bidirectional = True, we need to re-train the model with mask.
    2) inter_forward: frequency is reshaped to batch, so selecting the first k frequency bins will not hurt other frequency bins 

    for conventional-rnn, only inter_forward
3. use vad output (time-dimension) to determine the ratio of frequency bins
4. mid-fusion: genealize to different devices
'''
import torch
from .dpcrn import DPCRN
import torchaudio
    
class Skip_DPCRN(DPCRN):
    def __init__(self):
        super(Skip_DPCRN, self).__init__()
        self.threshold = 0.8
        self.comp_ratio = 1
    def skip_forward(self, x, acc, skip):
        '''
        x: [B, N, K, S] Batch, feature, frequency, time
        acc: [B, N, 1, S] Batch, feature, 1, time
        skip: [B, 1, S] Batch, 1, time
        we also have B = 1, (not-batching)
        '''
        # skip = skip > 0.5
        # skip[:, :, :5, :] = True  
        # skip[:, :, 5:, :] = False  
        # acc = acc * skip
        # x = x * skip
        # return self.forward(x, acc)
        return self.vad_forward_inference(x, acc, skip)
    
    def dynamic_forward(self, d):   
        score = torch.cumsum(torch.sum(d, dim=(1, 3), keepdim=True), dim=2)
        score = score / torch.max(score, dim=2, keepdim=True)[0]
        if self.training:
            dt = self.rnn_layer(d)
            d = dt * score + (1-score) * d
            return d
        else:
            score = score < self.threshold
            # score[:, :, :5, :] = True  
            # score[:, :, 5:, :] = False  
            dt = self.rnn_layer(d)
            d = torch.masked_scatter(dt, ~score, d)
            self.comp_ratio = torch.mean(score.float(), dim=(2, 3)).cpu().numpy()
            return d
    def vad_forward_inference(self, x, acc, vad=None):
        '''
           x: [B, N, K, S] Batch, feature, frequency, time
           out: [B, N, K, S]
           vad: [B, 1, S] Batch, 1, time
           we also have B = 1, (not-batching)
        '''
        self.comp_ratio = 1
        batch = x.shape[0]
        if batch == 1:
            # real inference
            edge = torch.diff(vad, dim=-1, append=torch.zeros_like(vad[..., :1]))
            out = torch.zeros_like(x)
            temp = 0
            for t in range(vad.shape[-1]):
                if edge[..., t] == 1:
                    temp = t
                elif edge[..., t] == -1:
                    # out[..., temp:t], score_mean = self.dynamic_forward_inference(x[..., temp:t], acc[..., temp:t])
                    out[..., temp:t] = self.forward(x[..., temp:t], acc[..., temp:t])
                    temp = t
            self.comp_ratio = vad.mean(dim=(1), keepdims=True).cpu().numpy()
        else:
            # fake inference (only for debugging) by default right now
            out, vad = self.vad_forward_train(x, acc, vad)
            # vad[vad > 0.5] = 1; vad[vad <= 0.5] = 0
            out = torch.masked_scatter(out, ~(vad.bool()), torch.zeros_like(x))
            self.comp_ratio *= vad.mean(dim=(2, 3)).cpu().numpy()
        return out