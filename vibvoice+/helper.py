import torch
from feature import stft, istft
from loss import Spectral_Loss, sisnr
from evaluation import eval
def train_epoch(model, sample, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device)
    vad = sample['vad'].to(device)
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(noisy, 640, 320, 640)
    optimizer.zero_grad()
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
    loss = sisnr(est_audio, clean.squeeze(1))
    loss += Spectral_Loss(est_mag, clean_mag, vad)    
    if discriminator is not None:
        predict_fake_metric = discriminator(clean_mag, est_mag)
        loss += 0.05 * torch.nn.functional.mse_loss(predict_fake_metric.flatten(),
                                                torch.ones(noisy.shape[0]).to(device).float())
    loss.backward() 
    optimizer.step()
    if discriminator is not None:
        discrim_loss_metric = calculate_discriminator_loss(discriminator=discriminator, clean_mag=clean_mag, clean_audio=clean, est_mag = est_mag, est_audio=est_audio)
        if discrim_loss_metric is not None:
            optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])
    return loss.item()

def test_epoch(model, sample, device='cuda'):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
    vad = sample['vad'].to(device)
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")    
    est_audio = est_audio.cpu().numpy()
    clean = clean.cpu().numpy().squeeze(1)
    return eval(clean, est_audio)

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
