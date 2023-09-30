import torch
from evaluation import eval
from feature import stft, istft
from loss import Spectral_Loss, sisnr
from model import batch_pesq
from librosa.filters import mel as librosa_mel_fn
mel = librosa_mel_fn(16000, 512, 80, 0, 8000)
mel_basis = torch.from_numpy(mel).float()
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C
def scale_align(clean, acc):
    clean_rms = (clean ** 2).mean(dim=-1, keepdim=True) ** 0.5
    acc_rms = (acc ** 2).mean(dim=-1, keepdim=True) ** 0.5
    snr_scalar = clean_rms / (acc_rms + 1e-6)
    acc *= snr_scalar
    return acc
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))
    y = torch.nn.functional.pad(y, (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    spec = torch.stft(y.squeeze(1), n_fft, hop_length=hop_size, win_length=win_size, window=torch.hann_window(win_size, device=y.device),
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis.to(y.device), spec)
    spec = dynamic_range_compression_torch(spec)
    spec = spec.unsqueeze(1)
    return spec

def train_DPCRN(model, sample, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); vad = sample['vad'].to(device)
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
def test_DPCRN(model, sample, device='cuda'):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); vad = sample['vad'].to(device)
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")    
    est_audio = est_audio.cpu().numpy()
    clean = clean.cpu().numpy().squeeze(1)
    return clean, est_audio

def train_TWO_STAGE(model, sample, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); vad = sample['vad'].to(device)
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
def test_TWO_STAGE(model, sample, device='cuda'):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); vad = sample['vad'].to(device)
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")    
    est_audio = est_audio.cpu().numpy()
    clean = clean.cpu().numpy().squeeze(1)
    return clean, est_audio


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
