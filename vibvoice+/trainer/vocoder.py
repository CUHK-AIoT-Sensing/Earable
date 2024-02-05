import torch
from feature import stft, istft, mel_filterbank, mel_spectrogram
from loss import get_loss, eval
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import numpy as np
import os
from .adversarial import calculate_discriminator_loss
def train_epoch(model, train_loader, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    Loss_list = []
    model.train()
    pbar = tqdm(train_loader)
    mel_basis, hann_window = mel_filterbank(1024, 80, 16000, 1024, 0, 8000, device)
    for sample in pbar:
        acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device)
        mel_clean = mel_spectrogram(clean, 1024, mel_basis, hann_window, 256, 1024,)
        mel_noisy = mel_spectrogram(noisy, 1024, mel_basis, hann_window, 256, 1024,)
        mel_acc = mel_spectrogram(acc, 1024, mel_basis, hann_window, 256, 1024,)
        optimizer.zero_grad()
        est_mel = model.analysis(mel_noisy, mel_acc)
        loss = torch.nn.functional.mse_loss(est_mel, mel_clean)    
        loss.backward() 
        optimizer.step()
        Loss_list.append(loss.item())
        pbar.set_description("loss: %.3f" % (loss.item()))
    return np.mean(Loss_list)

def test_epoch(model, dataset, BATCH_SIZE, device='cuda'):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    Metric = []
    model.eval()
    mel_basis, hann_window = mel_filterbank(1024, 80, 16000, 1024, 0, 8000, device)
    with torch.no_grad():
        for sample in pbar:
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
            mel_clean = mel_spectrogram(clean, 1024, mel_basis, hann_window, 256, 1024,)
            mel_noisy = mel_spectrogram(noisy, 1024, mel_basis, hann_window, 256, 1024,)
            mel_acc = mel_spectrogram(acc, 1024, mel_basis, hann_window, 256, 1024,)
            est_mel = model.analysis(mel_noisy, mel_acc)
            L1_loss = torch.nn.functional.l1_loss(est_mel, mel_clean).item()
            metric = [L1_loss]
            Metric.append(metric)  
    avg_metric = np.round(np.mean(Metric, axis=0), 2).tolist()
    print(avg_metric)
    return avg_metric
def test_epoch_save(model, dataset, dir, output_dir, device='cuda'):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    model.eval()
    mel_basis, hann_window = mel_filterbank(1024, 80, 16000, 1024, 0, 8000, device)

    with torch.no_grad():
        for sample in pbar:
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device)
            mel_noisy = mel_spectrogram(noisy, 1024, mel_basis, hann_window, 256, 1024,)
            mel_acc = mel_spectrogram(acc, 1024, mel_basis, hann_window, 256, 1024,)
            est_mel = model.analysis(mel_noisy, mel_acc)
            est_audio = model.generation(est_mel)
            print(est_audio.shape, est_audio.max(), est_audio.min())
            est_audio = est_audio.squeeze()
            est_audio = est_audio * 32768.0
            est_audio = est_audio.cpu().numpy().astype('int16')
            fname = sample['file'][0]
            fname = fname.replace(dir, output_dir)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            wavfile.write(fname, 16000, est_audio)
def train_epoch_tta(model, dataset, dir, output_dir, device='cuda'):
    return NotImplementedError