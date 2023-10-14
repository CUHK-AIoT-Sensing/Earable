'''
based on HGCN: HARMONIC GATED COMPENSATION NETWORK FOR SPEECH ENHANCEMENT and HGCN+
we mainly keep the harmonic matrix, the gated harmonic compensation 
'''
import torch.nn as nn
import torch
from .base_model import DPCRN_basic
import numpy as np
class CausalPool1d(nn.Module):
    def __init__(self, ker, str):
        super(CausalPool1d, self).__init__()
        self.smooth = nn.AvgPool1d(kernel_size=ker, stride=str, padding=0)
        self.left_pad = ker - 1

    def forward(self, x):
        x = torch.nn.functional.pad(x, [self.left_pad, 0], value=1e-8)
        return self.smooth(x)
class HarmonicIntegral(nn.Module):
    def __init__(self, corr_path, loc_path, harmonic_num=1):
        super(HarmonicIntegral, self).__init__()
        self.harmonic_smooth = CausalPool1d(ker=3, str=1)
        self.harmonic_num = harmonic_num
        if corr_path is not None:
            hi_integral_matrix = torch.tensor(np.load(corr_path), dtype=torch.float).unsqueeze(0).unsqueeze(0)
            harmonic_loc = torch.tensor(np.load(loc_path), dtype=torch.float).unsqueeze(0).unsqueeze(0)
        else:
            # for loading param
            hi_integral_matrix = torch.randn(1, 1, 4200, 321)
            harmonic_loc = torch.randn(1, 1, 4200, 321)
        self.register_buffer("integral_m", hi_integral_matrix)
        self.register_buffer("harmonic_loc", harmonic_loc)
        self.integral_m[self.integral_m != self.integral_m] = 0  # deal nan
        self.harmonic_loc[self.harmonic_loc != self.harmonic_loc] = 0

    def forward(self, mag, freq_dim=257):
        """
        :param x: B,2*C,F,T
        :param freq_dim:
        :return:
        """
        harmonic_nominee = torch.matmul(self.integral_m, mag)
        value, position = torch.topk(harmonic_nominee[:, :, :], k=self.harmonic_num, dim=-2)
        choosed_harmonic = torch.zeros(mag.size(0), mag.size(1), freq_dim, mag.size(-1)).to(mag.device)
        for i in range(self.harmonic_num):
            choose = self.harmonic_smooth(position.to(torch.float)[:, :, i, :]).to(torch.long)
            choosed_harmonic += self.harmonic_loc[:, :, choose, :][0][0].permute(0, 1, 3, 2)
        choosed_harmonic = (choosed_harmonic > 0).to(torch.float)
        return choosed_harmonic
class CausalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, bias=True):
        super(CausalConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.left_pad = kernel_size[1] - 1
        padding = (kernel_size[0] // 2, self.left_pad)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        B, C, F, T = x.size()
        return self.conv(x)[..., :T]
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, bias, out_activate):
        super(ResidualBlock, self).__init__()
        self.convblock = nn.Sequential(
            CausalConv(in_ch, out_ch, kernel_size, stride, bias),
            nn.PReLU(),
            CausalConv(out_ch, out_ch, kernel_size, stride, bias),
        )
        self.out_activate = out_activate

    def forward(self, x):
        out = self.convblock(x)
        out = self.out_activate(out + x)
        return out
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(1, 1)):
        super(GatedConv2d, self).__init__()
        gate_ch = 1
        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels + gate_ch),
            CausalConv(in_channels + gate_ch, in_channels + gate_ch, (1, 1), (1, 1), bias=True),
            nn.PReLU(),
            CausalConv(in_channels + gate_ch, 1, (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.out_conv = CausalConv(in_ch=in_channels, out_ch=out_channels, kernel_size=kernel_size, stride=stride,
                                   bias=False)

    def forward(self, input_features, gating_features):
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
        feature = input_features * alphas
        return self.out_conv(feature)
class GHCM(nn.Module):
    def __init__(self, inch=1, chs=(8, 16, 8)):
        super(GHCM, self).__init__()
        self.chs = (inch,) + chs + (1,)
        self.activate = [nn.PReLU() for _ in range(len(self.chs) - 2)]
        self.activate.append(nn.BatchNorm2d(1))
        self.body = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        for i in range(len(self.chs) - 1):
            self.body.append(
                nn.Sequential(
                    ResidualBlock(in_ch=self.chs[i + 1], out_ch=self.chs[i + 1], kernel_size=(5, 2), stride=(1, 1),
                                  bias=False, out_activate=self.activate[i])
                )
            )
            self.gate_convs.append(
                GatedConv2d(in_channels=self.chs[i], out_channels=self.chs[i + 1], kernel_size=(5, 2))
            )

    def forward(self, gate, in_feature):
        out = in_feature
        for index in range(len(self.body)):
            out = self.gate_convs[index](input_features=out, gating_features=gate)
            out = self.body[index](out)
        return out

    def bias_apply(self, x_origin, in_feature, mask_out):
        """
        :param mask_out: B,1,F,T
        :param in_feature: B,F,T
        :param x_origin: B,2*F,T
        :return:
        """
        mask_out = mask_out.squeeze(1)
        real, imag = torch.chunk(x_origin, 2, 1)
        real = real[:, 1:, :]
        imag = imag[:, 1:, :]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        real_phase = real / (spec_mags + 1e-8)
        imag_phase = imag / (spec_mags + 1e-8)
        s1_mag = self.mag(complexC2F(in_feature))
        bias_mag = torch.sigmoid(mask_out) * s1_mag
        est_mags = bias_mag + spec_mags
        real = F.pad(est_mags * real_phase, [0, 0, 1, 0], value=0)
        imag = F.pad(est_mags * imag_phase, [0, 0, 1, 0], value=0)
        return torch.cat([real, imag], 1)

    def mag(self, x):
        """
        :param x:B,2*F,T
        :return:
        """
        return torch.stack(torch.chunk(x, 2, dim=-2), dim=-1).pow(2).sum(dim=-1).sqrt()

class HGCN(nn.Module):
    def __init__(self, harmonic_num=1, gsrm_chs=(8, 16, 8),
                 corr_path="./harmonic_integral/harmonic_integrate_matrix.npy",
                 loc_path="./harmonic_integral/harmonic_loc.npy", train_flag=False):
        super(HGCN, self).__init__()
        self.train_flag = train_flag
        self.hi = HarmonicIntegral(corr_path=corr_path, loc_path=loc_path, harmonic_num=harmonic_num)
        self.cem = DPCRN_basic(last_channel=1)
        self.hm = torch.nn.GRU(321, 321)
        
        self.fc = nn.Linear(10, 1)
        self.ghcm = GHCM(inch=1, chs=gsrm_chs)
    def harmonic_mask(self, x):
        harmonic_loc = self.hi(x, freq_dim=321)[:, :, 1:, :]
        region_a = torch.argmax(region_a, -1).to(torch.float)
        region_a = region_a[:, 1:, :].unsqueeze(1)
        gate = region_a * harmonic_loc 
        return gate
    def forward(self, x, acc):

        cem_output = self.cem(acc)
        mask = torch.sigmoid(cem_output[:, :1, :, :])
        ca = cem_output[:, 1:, :, :]
        region_a = self.fc(ca)
        spec1 = mask * x
        with torch.no_grad():
            harmonic_loc = self.hi(spec1, freq_dim=321)[:, :, 1:, :]
            region_a = torch.argmax(region_a, -1).to(torch.float)
            region_a = region_a[:, 1:, :].unsqueeze(1)
            gate = region_a  * harmonic_loc 
        spec1 = self.ghcm(spec1) * gate

        # spec1 = spec1
        in_feature = complexF2C(spec1)[:, :, 1:, :]
        spec2 = self.ghcm(gate=gate, in_feature=in_feature, origin_spec=spec1)
        results.append(spec2)
        if self.train_flag:
            return results, regions
        else:
            return self.istft(results[-1]).squeeze(1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hi = HarmonicIntegral(corr_path="./harmonic_integral/harmonic_integrate_matrix.npy",loc_path="./harmonic_integral/harmonic_loc.npy")
    dummy_input = torch.zeros(1, 1, 321, 100)
    dummy_input[:, :, 30:32, :50] = 1
    dummy_input[:, :, 35:37, 50:] = 100
    output = hi(dummy_input, freq_dim=321)
    print(output.shape)
    plt.imshow(output[0][0].detach().numpy(), aspect='auto', origin='lower')
    plt.savefig("harmonic_loc.png")

    hgcn = HGCN()
    x = torch.randn(1, 1, 321, 100)
    acc = torch.randn(1, 1, 321, 100)
    output = hgcn(x, acc)