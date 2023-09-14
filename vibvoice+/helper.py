import torch
from evaluation import eval
from feature import stft, istft
from loss import Spectral_Loss
from model import batch_pesq
def decode(model, noisy_mag, noisy_phase, noisy_real, noisy_imag, acc, method='magnitude'):
    if method == 'magnitude':
        predict_mask = model(noisy_mag, acc)
        est_mag = predict_mask * noisy_mag
        est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
    elif method == 'real_imag':
        noisy_input = torch.cat([noisy_real, noisy_imag], dim=1)
        predict_mask = model(noisy_input, acc)
        predict = predict_mask * noisy_input
        est_mag = torch.abs(torch.complex(predict[:, :1], predict[:, 1:]))
        est_audio = istft((predict[:, 0], predict[:, 1]), 640, 320, 640, input_type="real_imag")
    else: # mag_real_imag
        noisy_input = torch.cat([noisy_mag, noisy_real, noisy_imag], dim=1)
        predict = model(noisy_input, acc)
        est_real = torch.sigmoid(predict[:, :1]) * noisy_real
        est_imag = torch.sigmoid(predict[:, :1]) * noisy_imag
        phase_square = (predict[:, 1:2]**2 + predict[:, 2:]**2) ** 0.5
        phase_sin = predict[:, 1:2] / phase_square; phase_cos = predict[:, 1:2]/ phase_square
        est_real = est_real * phase_cos - est_imag * phase_sin
        est_imag = est_real * phase_sin + est_imag * phase_cos
        est_mag = torch.abs(torch.complex(est_real, est_imag))
        est_audio = istft((est_real.squeeze(1), est_imag.squeeze(1)), 640, 320, 640, input_type="real_imag")
    return est_mag, est_audio
def train_DPCRN(model, sample, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); vad = sample['vad']
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    optimizer.zero_grad()
    acc, _, _, _ = stft(acc, 640, 320, 640)

    est_mag, est_audio = decode(model, noisy_mag, noisy_phase, noisy_real, noisy_imag, acc, method='magnitude')
    loss = Spectral_Loss(est_mag, clean_mag)
    loss += 0.2 * torch.mean(torch.abs(est_audio - clean)) 

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
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); vad = sample['vad']
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag, est_audio = decode(model, noisy_mag, noisy_phase, noisy_real, noisy_imag, acc, method='magnitude')
    est_audio = est_audio.cpu().numpy()
    clean = clean.cpu().numpy().squeeze(1)
    return clean, est_audio

def train_SUPER(model, sample, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); vad = sample['vad']
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    optimizer.zero_grad()
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag, est_audio = decode(model, noisy_mag, noisy_phase, noisy_real, noisy_imag, acc, method='magnitude')
    loss = Spectral_Loss(est_mag, clean_mag)
    loss += 0.2 * torch.mean(torch.abs(est_audio - clean)) 
    loss.backward() 
    optimizer.step()
    return loss.item()
def test_SUPER(model, sample, device='cuda'):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); vad = sample['vad']
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag, est_audio = decode(model, noisy_mag, noisy_phase, noisy_real, noisy_imag, acc, method='magnitude')
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

def train_PULSE(model, sample, optimizer, device='cuda'):
    optimizer.zero_grad()
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, _, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    # acc = torch.mean(acc, dim=1, keepdim=True)
    vad = vad.reshape(vad.shape[0], 1, 1, -1)

    # mask, ratio = get_mask(acc, vad)
    mask = sample['mask'].reshape(-1, 1, 1, 1) * torch.ones(noisy_mag.size(), dtype = torch.float32); ratio = 0.6

    predict_mask = model(noisy_mag.to(device), acc.to(device))
    loss = weighted_pu_loss(mask.to(device), predict_mask, noisy_mag.to(device), beta=0, gamma=0.096, prior=ratio, p=1.0, mode='nn')
    loss.backward()
    optimizer.step()
    return loss.item()

def test_PULSE(model, sample, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    acc = torch.norm(acc, dim=1, p=2, keepdim=True)
    predict_mask = model(noisy_mag.to(device), acc.to(device))
    predict = (torch.sigmoid(-predict_mask).cpu() * noisy_mag)
    predict = istft((predict.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy().squeeze(1)
    return clean, predict

def train_MIXIT(model, sample, optimizer, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; 
    vad = sample['vad']; noise = sample['noise']; mixture = sample['mixture']

    mixture_mag, _, _, _ = stft(mixture, 640, 320, 640)
    noisy_mag, _, _, _ = stft(noisy, 640, 320, 640)
    noise_mag, _, _, _ = stft(noise, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    vad = vad.reshape(vad.shape[0], 1, 1, -1)

    optimizer.zero_grad()
    predict_mask = model(mixture_mag.to(device) ** 1/15, acc.to(device))
    loss = MixIT_loss(predict_mask, mixture_mag.to(device), noise_mag.to(device), noisy_mag.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def test_MIXIT(model, sample, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']; 

    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    vad = vad.reshape(vad.shape[0], 1, 1, -1)

    predict_mask = model(noisy_mag.to(device) ** 1/15, acc.to(device))
    predict = noisy_mag * torch.sigmoid(predict_mask[:, :1, :, :]).cpu()
    predict = istft((predict.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy().squeeze(1)
    return clean, predict

def train_Skip_DPCRN(model, sample, optimizer, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, _, _, _ = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    acc = torch.norm(acc.to(device=device), dim=1, p=2, keepdim=True)
    vad = vad.to(device=device).reshape(vad.shape[0], 1, 1, -1)

    predict_mask, predict_vad = model.vad_forward_train(noisy_mag, acc)
    # loss_vad = torch.nn.functional.binary_cross_entropy(predict_vad.squeeze(), vad.to(device=device))
    loss_se = Spectral_Loss(predict_mask * noisy_mag, clean_mag)
    loss = loss_se * 1
    loss.backward()
    optimizer.step()
    return loss.item()

def test_Skip_DPCRN(model, sample, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    acc = torch.norm(acc.to(device=device), dim=1, p=2, keepdim=True)
    vad = vad.to(device=device).reshape(vad.shape[0], 1, 1, -1)
    
    predict_mask = model.vad_forward_inference(noisy_mag, acc, vad)
    predict = (predict_mask * noisy_mag).cpu()
    predict = istft((predict.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy().squeeze(1)
    return clean, predict
