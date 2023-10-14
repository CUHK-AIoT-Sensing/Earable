'''
We summarize all the two-stage methods:
1. Harmonic Gated compensation network 
2. Deepfilternet
3. DMF-Net
'''
import torch.nn as nn
import torch
from .base_model import DPCRN_basic
from .hgcn import HarmonicIntegral, GHCM
class TWO_STAGE(nn.Module):
    def __init__(self, model):
        super(TWO_STAGE).__init__()
        if model == 'HGCN':
            self.model = HGCN()
        elif model == 'Deepfilternet':
            self.model = DeepFilterNet()
        else:
            self.model = DMF()
    def forward(self, x, acc, vad=1):
        return self.model(x, acc, vad)        
class HGCN(nn.Module):
    def __init__(self,):
        super(HGCN, self).__init__()
        corr_path = "./model/harmonic_integral/harmonic_integrate_matrix.npy"
        loc_path = "./model/harmonic_integral/harmonic_loc.npy"

        self.band = 65
        self.cem = DPCRN_basic(channel_list=[16, 32, 64, 128], init_channel=2, pad=[2])
        # self.hb = DPCRN_basic(channel_list=[16, 32, 64], init_channel=1, pad=[2])
        self.hi = HarmonicIntegral(corr_path=corr_path, loc_path=loc_path, harmonic_num=1)
        self.ghcm = GHCM(inch=1, chs=(8, 16, 8))

    def forward(self, x, acc, vad=1):

        # x, x_h = x[:, :, :self.band, :], x[:, :, self.band:, :]
        # acc, acc_h = acc[:, :, :self.band, :], acc[:, :, self.band:, :]
        # hb_out = torch.sigmoid(self.hb(x_h)) * x_h

        # coarse_mask = torch.sigmoid(self.cem(torch.cat([x, acc], dim=1)))
        # coarse_output = coarse_mask * x
        # coarse_output = torch.cat([coarse_output, hb_out], dim=2)
        # harmonic_loc = self.hi(coarse_output, freq_dim=321)[:, :, :self.band, :]
        # gate = vad.unsqueeze(2) * harmonic_loc
        # fine_output = torch.sigmoid(self.ghcm(gate=gate, in_feature=coarse_output[:, :, :self.band, :])) * x + coarse_output 
        # fine_output = torch.cat([fine_output, hb_out], dim=2)

        coarse_mask = torch.sigmoid(self.cem(torch.cat([x, acc], dim=1)))
        coarse_output = coarse_mask * x
        harmonic_loc = self.hi(coarse_output, freq_dim=321)
        gate = vad.unsqueeze(2) * harmonic_loc
        fine_output = torch.sigmoid(self.ghcm(gate=gate, in_feature=coarse_output)) * x + coarse_output 
        return coarse_output, fine_output
    
class DeepFilterNet(nn.Module):
    def __init__(self,):
        super(DeepFilterNet, self).__init__()
    
    def forward(self, x, acc, vad=1):

        return 

class DMF(nn.Module):
    def __init__(self,):
        super(DMF, self).__init__()
        self.LF = DPCRN_basic(channel_list=[16, 32, 64, 128, 256], init_channel=2, pad=[3])
        self.MF = DPCRN_basic(channel_list=[16, 32, 64, 128], init_channel=2, pad=[3])
        self.HF = DPCRN_basic(channel_list=[16, 32, 64], init_channel=3, pad=[2])
        self.band = [[0, 65], [65, 193], [193, 321]]
    def forward(self, x, acc):
        input_LF = torch.cat([x[:, :, self.band[0][0]:self.band[0][1], :], acc[:, :, self.band[0][0]:self.band[0][1], :]], dim=1)
        out_LF = torch.sigmoid(self.LF(input_LF))
        pad_out_LF = x[:, :, self.band[0][0]:self.band[0][1], :] * out_LF
        pad_out_LF = torch.nn.functional.pad(out_LF, (0, 0, 0, 64, ), value=0)[:, :, 1:, :]

        input_MF = torch.cat([pad_out_LF, x[:, :, self.band[1][0]:self.band[1][1], :]], dim=1)    
        out_MF = torch.sigmoid(self.MF(input_MF))
        pad_out_MF = x[:, :, self.band[1][0]:self.band[1][1], :] * out_MF

        input_HF = torch.cat([pad_out_LF, pad_out_MF, x[:, :, self.band[2][0]:self.band[2][1], :]], dim=1)  
        out_HF = torch.sigmoid(self.HF(input_HF))

        mask = torch.cat([out_LF, out_MF, out_HF], dim=2)
        return mask * x