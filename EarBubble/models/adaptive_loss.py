
from asteroid.losses.sdr import SingleSrcNegSDR
import torch 
import torch.nn as nn
class LogPowerLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)

        return: (B)
        """
        B, C, T = est.shape

        assert torch.abs(gt).max() < 1e-6, "This loss must only be used when gt = 0"
        est = est.reshape(B*C, T) # [BC, T]
        loss = 10 * torch.log10(torch.sum(est ** 2, axis=-1) + 1e-3)
        loss = loss.reshape(B, C) # [B, C]
        loss = loss.mean(axis=-1) # [B]
        return loss

class SNRLosses(nn.Module):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.name = name
        if name == 'sisdr':
            self.loss_fn = SingleSrcNegSDR('sisdr')
        elif name == 'snr':
            self.loss_fn = SingleSrcNegSDR('snr')
        # elif name == 'sdsdr':
        #     self.loss_fn = SingleSrcNegSDR('sdsdr')
        elif name == 'fused':
            self.loss1 = SingleSrcNegSDR('sisdr')
            self.loss2 = SingleSrcNegSDR('snr')
        elif name == "max_fused":
            self.loss1 = SingleSrcNegSDR('sisdr')
            self.loss2 = SingleSrcNegSDR('snr')
        elif name == "sdsdr":
            self.loss1 = SingleSrcNegSDR("snr")
            self.loss2 = SingleSrcNegSDR("sdsdr")
        elif name == "full":
            self.loss1 = SingleSrcNegSDR("snr")
            self.loss2 = SingleSrcNegSDR("sdsdr")
            self.loss3 = SingleSrcNegSDR('sisdr')
        else:
            assert 0, f"Invalid loss function used: Loss {name} not found"

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        B, C, T = est.shape

        est = est.reshape(B*C, T)
        gt = gt.reshape(B*C, T)
        
        if self.name == "fused":
            return 0.5*self.loss1(est_target=est, target=gt) + 0.5*self.loss2(est_target=est, target=gt)
        elif self.name == "max_fused" or self.name == "sdsdr":
            return torch.maximum(self.loss1(est_target=est, target=gt), self.loss2(est_target=est, target=gt)  )
        elif self.name == "full":
            l1 = self.loss1(est_target=est, target=gt)
            l2 = self.loss2(est_target=est, target=gt)
            l3 = self.loss3(est_target=est, target=gt)

            return  0.5*l3 + 0.5* torch.maximum(l1, l2)
        else:
            return self.loss_fn(est_target=est, target=gt)

class SNRLPLoss(nn.Module):
    def __init__(self, snr_loss_name = "snr") -> None:
        super().__init__()
        self.snr_loss = SNRLosses(snr_loss_name)
        # self.lp_loss = LogPowerLoss()
        # self.lp_loss = nn.L1Loss()#LogPowerLoss()
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor, neg_weight=0):
        """
        input: (B, C, T) (B, C, T)
        """
        comp_loss = torch.zeros((est.shape[0] * est.shape[1]), device=est.device)

        sample_mask = (torch.max(torch.abs(gt), dim=2)[0] == 0).reshape(-1)
        mask = (torch.max(torch.max(torch.abs(gt), dim=2)[0], dim=1)[0] == 0)
        # If there's at least one negative sample
        if any(sample_mask):
            neg_loss = torch.abs(est - gt).mean(dim=2).reshape(-1)
            # neg_loss = 10 * torch.log10(torch.sum(est ** 2, axis=-1) + 1e-3).reshape(-1)
            comp_loss[sample_mask] = neg_loss[sample_mask] * neg_weight
            
        # If there's at least one positive sample
        if any((~ sample_mask)):
            pos_loss = self.snr_loss(est, gt)
            # Compute_joint_loss
            comp_loss[~sample_mask] = pos_loss[~sample_mask]
        if self.training:
            return comp_loss.mean()
        else:
            return comp_loss[~sample_mask].mean(), comp_loss[sample_mask].mean(), 
