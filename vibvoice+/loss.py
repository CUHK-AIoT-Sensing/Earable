import numpy as np
import torch
from pesq import pesq, pesq_batch, PesqError
from joblib import Parallel, delayed
from pystoi.stoi import stoi
import numpy as np

EPS = 1e-12
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
def get_loss(est_audio, reference, vad=1):
    loss = 0
    # vad_time = vad.repeat(1, 1, 320).reshape(vad.shape[0], -1)
    # vad_time = vad_time[:, 320:-320]
    loss += sisnr(est_audio, reference, vad) * 0.9
    loss += snr(est_audio, reference, vad) * 0.1
    loss += MultiResolutionSTFTLoss(est_audio, reference, vad) 
    return loss
def eval(clean, predict):
    metrics = []
    clean_npy = clean.cpu().numpy()
    predict_npy = predict.cpu().numpy()
    pesq_wb = pesq_batch(16000, clean_npy, predict_npy, 'wb', n_processor=4, on_error=PesqError.RETURN_VALUES)
    pesq_nb = pesq_batch(16000, clean_npy, predict_npy, 'nb', n_processor=4, on_error=PesqError.RETURN_VALUES)
    metrics += [np.mean(pesq_wb), np.mean(pesq_nb)]

    stoi = np.mean(batch_stoi(clean_npy, predict_npy))
    metrics += [stoi]
    metrics.append(-snr(predict, clean).item())
    metrics.append(-sisnr(predict, clean).item())
    metrics.append(lsd(predict, clean).item())
    return metrics

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
def MultiResolutionSTFTLoss(x, y, vad=1):
    fft_size = [512, 1024, 2048,]
    hop_size = [60, 120, 240,]
    win_length = [300, 600, 1200, ]
    loss = 0
    for fft, hop, win in zip(fft_size, hop_size, win_length):
        window = torch.hann_window(win).to(x.device)
        x_stft = torch.stft(x, fft, hop, win, window, return_complex=True)
        x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=1e-8))
        y_stft = torch.stft(y, fft, hop, win, window, return_complex=True)
        y_mag = torch.sqrt(torch.clamp((y_stft.real**2) + (y_stft.imag**2), min=1e-8))
        loss += Spectral_Loss(x_mag, y_mag, vad)
    return loss
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
def snr(x, s, eps=1e-8, vad=1):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          snr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    return -20 * torch.log10(eps + l2norm(s_zm * vad) / (l2norm((x_zm - s_zm)*vad) + eps)).mean()
def lsd(x, s, eps=1e-8, vad=1):
    window = torch.hann_window(256).to(x.device)
    x_stft = torch.stft(x, 256, 120, 256, window, return_complex=True)
    x_mag = torch.sqrt(
        torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=1e-8)
    )
    s_stft = torch.stft(s, 256, 120, 256, window, return_complex=True)
    s_mag = torch.sqrt(
        torch.clamp((s_stft.real**2) + (s_stft.imag**2), min=1e-8)
    )
    lsd = torch.log10(x_mag **2 / ((s_mag + eps) ** 2) + eps) ** 2 * vad
    lsd = torch.mean(torch.mean(torch.mean(lsd, axis=1) ** 0.5, axis=-1))
    return lsd
def rmse(x, s, eps=1e-8, vad=1):
    return torch.mean(torch.sqrt(torch.mean(((x - s) * vad) ** 2, axis=-1)))


def batch_stoi(clean, noisy):
    stoi_score = Parallel(n_jobs=-1)(delayed(stoi)(c, n, 16000) for c, n in zip(clean, noisy))
    stoi_score = np.array(stoi_score)
    return stoi_score


