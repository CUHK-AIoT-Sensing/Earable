import torch
from tqdm import tqdm
from feature import istft, stft
import os
import scipy.io.wavfile as wavfile
from .helper import batching, unbatching, data_purification, Remix, update_teacher
from loss import get_loss
def BN_adapt(model, dataset, dir, output_dir, device):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    batch = {}; batch_count = []; length_count = []
    for sample in pbar:
        batch, batch_count, length_count = batching(sample, batch, batch_count, length_count, length=5, sr=16000)
        if batch['noisy'].shape[0] >= 16:
            model.train()
            sample = batch
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            with torch.no_grad():
                est_mag = model(noisy_mag, acc)
            est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").cpu().numpy()
            est_audio = unbatching(est_audio, batch_count, length_count)
            for i, est in enumerate(est_audio):
                fname = sample['file'][i]
                fname = fname.replace(dir, output_dir)
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                wavfile.write(fname, 16000, est[0])
            batch = {}; batch_count = []; length_count = []

def Remix_snr(model, dataset, dir, output_dir, device):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    pbar = tqdm(test_loader)
    batch = {}; batch_count = []; length_count = []
    model.train()
    for sample in pbar:
        batch, batch_count, length_count = batching(sample, batch, batch_count, length_count, length=5, sr=16000)
        if batch['noisy'].shape[0] >= 16:
            sample = batch
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); 
            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            sample, segsnr = data_purification(noisy_mag, acc, sample)

            optimizer.zero_grad()
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device)
            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            est_mag = model(noisy_mag, acc)
            est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
            loss = get_loss(est_audio, clean.squeeze(1), segsnr)
            loss.backward()
            optimizer.step()
            pbar.set_description("loss: %.3f" % (loss.item()))

            est_audio = unbatching(est_audio, batch_count, length_count).cpu().numpy()
            for i, est in enumerate(est_audio):
                fname = sample['file'][i]
                fname = fname.replace(dir, output_dir)
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                wavfile.write(fname, 16000, est[0])
            batch = {}; batch_count = []; length_count = []

def Remix_teacher(model, dataset, dir, output_dir, device):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    batch = {}; batch_count = []; length_count = []
    real_batch_count = 0
    import copy
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for sample in pbar:
        batch, batch_count = batching(est_audio, batch_count, length_count, length=5, sr=16000)
        if batch.shape[0] > 16:
            model.train()
            sample = batch
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            sample = Remix(noisy_mag, acc, noisy_phase, noisy, teacher_model, sample)
            
            optimizer.zero_grad()
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device)
            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            est_mag = model(noisy_mag, acc)
            est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
            loss = get_loss(est_audio, clean.squeeze(1))
            loss.backward()
            optimizer.step()
            pbar.set_description("loss: %.3f" % (loss.item()))

            real_batch_count += 1
            if real_batch_count % 8 == 0:
                teacher_model = update_teacher(model, teacher_model)
                
            est_audio = unbatching(est_audio, batch_count, length_count).cpu().numpy()
            for est in est_audio:
                fname = sample['file'][0]
                fname = fname.replace(dir, output_dir)
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                wavfile.write(fname, 16000, est[0])
            batch = {}; batch_count = []; length_count = []

