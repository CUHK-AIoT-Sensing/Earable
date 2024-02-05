import torch
from feature import stft, istft
from loss import get_loss, eval
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import numpy as np
import os
from .masker_tta import BN_adapt, Remix_snr, Remix_teacher
from .adversarial import calculate_discriminator_loss
def train_epoch(model, train_loader, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    Loss_list = []
    model.train()
    pbar = tqdm(train_loader)
    for sample in pbar:
        acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device)
        noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
        clean_mag, _, _, _ = stft(noisy, 640, 320, 640)
        optimizer.zero_grad()
        acc, _, _, _ = stft(acc, 640, 320, 640)
        est_mag = model(noisy_mag, acc)
        est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
        loss = get_loss(est_audio, clean.squeeze(1))
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
        Loss_list.append(loss.item())
        pbar.set_description("loss: %.3f" % (loss.item()))
    return np.mean(Loss_list)

def test_epoch(model, dataset, BATCH_SIZE, device='cuda'):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    Metric = []
    model.eval()
    with torch.no_grad():
        for sample in pbar:
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            est_mag = model(noisy_mag, acc)
            est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")    
            metric = eval(clean.squeeze(1), est_audio)
            Metric.append(metric)  
    avg_metric = np.round(np.mean(Metric, axis=0), 2).tolist()
    print(avg_metric)
    return avg_metric

def test_epoch_save(model, dataset, dir, output_dir, device='cuda'):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    model.eval()
    with torch.no_grad():
        for sample in pbar:
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            est_mag = model(noisy_mag, acc)
            est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").cpu().numpy()
            fname = sample['file'][0]
            fname = fname.replace(dir, output_dir)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            wavfile.write(fname, 16000, est_audio[0])

def train_epoch_tta(model, dataset, dir, output_dir, device='cuda', method='BN_adapt'):
    if method=='BN_adapt':
        print('tta method: BN_adapt')
        BN_adapt(model, dataset, dir, output_dir, device)
    elif method == 'Remix_snr':
        print('tta method: Remix_snr')
        Remix_snr(model, dataset, dir, output_dir, device)
    elif method == 'Remix_teacher':
        print('tta method: Remix_teacher')
        Remix_teacher(model, dataset, dir, output_dir, device)
    else:
        return NotImplementedError