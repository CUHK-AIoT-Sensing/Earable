import torch
import itertools

def get_mask(acc, vad):
    '''
    1. 
    noise -> inactivity -> mask = 1 (determined by ~vad)
    the others (unlabelled) -> mask = 2
    2. 
    acc -> activity -> mask of 1
    others (include nothing, noise, speech) -> inactivity -> all 0
    '''
    mask = torch.zeros_like(acc)
    mask = torch.masked_fill(mask, ~vad.bool(), 1)
    ratio = 1 - torch.mean(vad)
    # threshold = filters.threshold_otsu(acc.numpy())
    # mask = (acc > threshold).to(dtype=torch.float32)
    return mask, ratio
def Spectral_Loss(x_mag, y_mag, vad=1):
    """Calculate forward propagation.
          Args:
              x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
              y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
              vad (Tensor): VAD of groundtruth signal (B, #frames, #freq_bins).
          Returns:
              Tensor: Spectral convergence loss value.
          """
    x_mag = torch.clamp(x_mag, min=1e-7)
    y_mag = torch.clamp(y_mag, min=1e-7)
    spectral_convergenge_loss =  torch.norm(vad * (y_mag - x_mag), p="fro") / torch.norm(y_mag, p="fro")
    log_stft_magnitude = (vad * (torch.log(y_mag) - torch.log(x_mag))).abs().mean()
    return 0.5 * spectral_convergenge_loss + 0.5 * log_stft_magnitude
def sisnr(x, s, eps=1e-8, vad=1):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return -20 * torch.log10(eps + l2norm(t * vad) / (l2norm((x_zm - t)*vad) + eps)).mean()