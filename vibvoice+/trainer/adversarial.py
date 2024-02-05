import torch
import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score
def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy)
    )
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to("cuda")

def calculate_discriminator_loss(discriminator, clean_mag, clean_audio, est_mag, est_audio, ):
    pesq_score = batch_pesq(clean_audio.squeeze(1).cpu().numpy(), est_audio.detach().cpu().numpy())
    # The calculation of PESQ can be None due to silent part
    if pesq_score is not None:
        predict_enhance_metric = discriminator(
            clean_mag, est_mag.detach()
        )
        predict_max_metric = discriminator(
           clean_mag, clean_mag
        )
        discrim_loss_metric = torch.nn.functional.mse_loss(
            predict_max_metric.flatten(),  torch.ones(est_mag.shape[0]).to(est_mag.device))
        + torch.nn.functional.mse_loss(predict_enhance_metric.flatten(), pesq_score)
    else:
        discrim_loss_metric = None
    return discrim_loss_metric
