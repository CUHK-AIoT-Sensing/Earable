import numpy as np
import torch
from pesq import pesq, pesq_batch
from joblib import Parallel, delayed
from pystoi.stoi import stoi

def eval(clean, predict):
    metrics = []
    if len(clean.shape) == 3:
        metrics.append(MAE(clean, predict))
    else:
        metrics.append(batch_pesq(clean, predict, 'wb'))
        metrics.append(batch_pesq(clean, predict, 'nb'))
        metrics.append(SI_SDR(clean, predict))
        metrics.append(batch_stoi(clean, predict))
    return np.stack(metrics, axis=1)

def SI_SDR(reference, estimation, sr=16000):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    References
        SDR– Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy

    projection = optimal_scaling * reference

    noise = estimation - projection
    ratio = np.sum(projection ** 2, axis=-1) / (np.sum(noise ** 2, axis=-1) + 1e-6)
    return 10 * np.log10(ratio)

def LSD(target, est):
    lsd = torch.log10(target**2 / ((est + 1e-8) ** 2) + 1e-8) ** 2
    lsd = torch.mean(torch.mean(lsd, dim=1) ** 0.5, dim=1)
    return lsd

def MAE(clean, predict):
    error = np.abs(clean - predict).mean(axis=(1, 2))
    return error

def batch_pesq(clean, noisy, mode):
    pesq_score = Parallel(n_jobs=-1)(delayed(pesq)(16000, c, n, mode, on_error=1) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    return pesq_score

def batch_stoi(clean, noisy):
    stoi_score = Parallel(n_jobs=-1)(delayed(stoi)(c, n, 16000) for c, n in zip(clean, noisy))
    stoi_score = np.array(stoi_score)
    return stoi_score


