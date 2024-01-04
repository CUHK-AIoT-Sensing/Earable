import torch
from feature import stft, istft
from loss import get_loss, eval
from tqdm import tqdm
import numpy as np
def train_epoch(model, train_loader, optimizer, device='cuda',):
    Loss_list = []
    model.train()
    pbar = tqdm(train_loader)
    for sample in pbar:
        acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); raw = sample['raw'].to(device)

        optimizer.zero_grad()
        noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
        acc, _, _, _ = stft(acc, 640, 320, 640)
        est_mag = model(noisy_mag, acc)
        est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
        loss = get_loss(est_audio, raw.squeeze(1))
        loss.backward() 
        optimizer.step()
        Loss_list.append(loss.item())
        pbar.set_description("loss: %.3f" % (loss.item()))
    return np.mean(Loss_list)

def test_epoch(model, dataset, BATCH_SIZE, device='cuda'):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=2, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    Metric = []
    model.eval()
    with torch.no_grad():
        for sample in pbar:
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); raw = sample['raw'].to(device)

            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            est_mag = model(noisy_mag, acc)
            est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")    
            metric = eval(raw, est_audio)
            Metric.append(metric)  
    avg_metric = np.round(np.mean(np.concatenate(Metric, axis=0), axis=0),2).tolist()
    print(avg_metric)
    return avg_metric
