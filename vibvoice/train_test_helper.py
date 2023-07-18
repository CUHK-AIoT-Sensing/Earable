import torch
import numpy as np
from evaluation import batch_pesq, SI_SDR, batch_stoi, eval_ASR, LSD
import torch.nn.functional as F
from scipy import signal
from mask import build_complex_ideal_ratio_mask, decompress_cIRM
from feature import stft, istft
# from speechbrain.pretrained import EncoderDecoderASR
'''
This script contains 4 model's training and test due to their large differences (for concise)
1. FullSubNet, LSTM, spectrogram magnitude -> cIRM
2. SEANet, GAN, time-domain, encoder-decoder,
3. A2Net (VibVoice), spectrogram magnitude -> spectrogram magnitude
4. Conformer, GAN, spectrogram real+imag -> real+imag
'''
# Uncomment for using another pre-trained model
# asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
#                                            savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
#  
#                                         run_opts={"device": "cuda"})

freq_bin_high = 33
transfer_function = np.load('function_pool.npy')
length_transfer_function = transfer_function.shape[0]
def OverlapAndAdd(inputs,frame_shift):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = np.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype)
        ones = np.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones
def synthetic(clean):
    time_bin = clean.shape[-1]
    index = np.random.randint(0, length_transfer_function)
    f = transfer_function[index, :, 0]
    v = transfer_function[index, :, 1]
    response = np.tile(np.expand_dims(f, axis=1), (1, time_bin))
    for j in range(time_bin):
        response[:, j] += np.random.normal(0, v, (freq_bin_high))
    acc = torch.from_numpy(response / np.max(f)).to(clean.device) * clean[:, :freq_bin_high, :]
    acc = clean[:, :freq_bin_high, :]
    return acc
def eval(clean, predict, text=None):
    if text is not None:
        wer_clean, wer_noisy = eval_ASR(clean, predict, text, asr_model)
        metrics = [wer_clean, wer_noisy]
    else:
        metric1 = batch_pesq(clean, predict, 'wb')
        metric2 = batch_pesq(clean, predict, 'nb')
        metric3 = SI_SDR(clean, predict)
        metric4 = batch_stoi(clean, predict)
        metrics = [metric1, metric2, metric3, metric4]
    return np.stack(metrics, axis=1)

def Spectral_Loss(x_mag, y_mag):
    """Calculate forward propagation.
          Args:
              x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
              y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
          Returns:
              Tensor: Spectral convergence loss value.
          """
    x_mag = torch.clamp(x_mag, min=1e-7)
    y_mag = torch.clamp(y_mag, min=1e-7)
    spectral_convergenge_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
    log_stft_magnitude = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
    return 0.5 * spectral_convergenge_loss + 0.5 * log_stft_magnitude
def train_TCNN(model, acc, noise, clean, optimizer, device='cuda'):
    optimizer.zero_grad()
    predict = model(noise.to(device).unsqueeze(1))
    #spec_predict = torch.stft(predict, 640, 320, 640, window=torch.hann_window(640, device=predict.device), return_complex=True).abs()
    #spec_clean = torch.stft(clean, 640, 320, 640, window=torch.hann_window(640, device=clean.device), return_complex=True).abs().to(device)
    # loss = Spectral_Loss(spec_clean, spec_predict)
    loss = F.mse_loss(predict.squeeze(1), clean.to(device).unfold(-1, 640, 320))
    loss.backward()
    optimizer.step()
    return loss.item()
def test_TCNN(model, acc, noise, clean, device='cuda', text=None):
    predict = model(noise.to(device).unsqueeze(1))
    predict = predict.cpu().squeeze(1).numpy()
    predict = OverlapAndAdd(predict, 320)
    clean = clean.numpy()
    return eval(clean, predict, text=text)
def train_CRN(model, acc, noise, clean, optimizer, device='cuda'):
    noisy_mag, _, _, _ = stft(noise, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    optimizer.zero_grad()
    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    if acc == None:
        acc = synthetic(clean_mag)
    else:
        acc, _, _, _ = stft(acc, 64, 32, 64)
        acc = torch.norm(acc.to(device=device), dim=1, p=2)
    predict_clean, predict_acc = model(noisy_mag, acc)
    loss1 = Spectral_Loss(predict_clean, clean_mag.unsqueeze(1))
    loss2 = 0.2 * Spectral_Loss(predict_acc, clean_mag.unsqueeze(1)[:, :, :freq_bin_high, :])
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    return loss.item()
def test_CRN(model, acc, noise, clean, device='cuda', text=None):
    noisy_mag, noisy_phase, _, _ = stft(noise, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    if acc == None:
        acc = synthetic(clean_mag)
    else:
        acc, _, _, _ = stft(acc, 64, 32, 64)
        acc = torch.norm(acc.to(device=device), dim=1, p=2)

    predict, _ = model(noisy_mag, acc)
    predict = predict.cpu().squeeze(1)
    predict = istft((predict, noisy_phase), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy()
    return eval(clean, predict, text=text)

def train_vibvoice(model, acc, noise, clean, optimizer, device='cuda'):
    noisy_mag, _, _, _ = stft(noise, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    optimizer.zero_grad()
    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    predict, acc = model(noisy_mag, acc)
    loss = Spectral_Loss(torch.squeeze(predict, 1), clean_mag[:, 1:257, 1:])
    loss += 0.1 * F.mse_loss(torch.squeeze(acc, 1), clean_mag[:, :32, 1:])
    loss.backward()
    optimizer.step()
    return loss.item()
def test_vibvoice(model, acc, noise, clean, device='cuda', text=None):
    noisy_mag, noisy_phase, _, _ = stft(noise, 640, 320, 640)
    noisy_mag = noisy_mag.to(device=device)
    predict = model(noisy_mag, acc)
    if isinstance(predict, tuple):
        predict = predict[0]
    predict = predict.cpu()
    predict = F.pad(predict, (1, 0, 1, 321 - 257)).squeeze(1)
    predict = istft((predict, noisy_phase), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy()
    return eval(clean, predict, text=text)

def train_fullsubnet(model, acc, noise, clean, optimizer, device='cuda'):
    optimizer.zero_grad()
    noise = noise.to(device=device)
    clean = clean.to(device=device)

    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noise, 512, 256, 512)
    _, _, clean_real, clean_imag = stft(clean, 512, 256, 512)
    cIRM = build_complex_ideal_ratio_mask(noisy_real.unsqueeze(1), noisy_imag.unsqueeze(1),
                                          clean_real.unsqueeze(1), clean_imag.unsqueeze(1))  # [B, F, T, 2]
    cIRM = cIRM.squeeze(1).permute((0, 3, 1, 2))

    noisy_mag = noisy_mag.unsqueeze(1)
    cRM = model(noisy_mag)
    loss = F.mse_loss(cIRM, cRM)

    loss.backward()
    optimizer.step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    return loss.item()

def test_fullsubnet(model, acc, noise, clean, device='cuda', text=None, data=False):
    noisy_mag, _, noisy_real, noisy_imag = stft(noise, 512, 256, 512)
    noisy_mag = noisy_mag.to(device=device).unsqueeze(1)
    predict = model(noisy_mag)
    cRM = decompress_cIRM(predict.permute(0, 2, 3, 1)).cpu()
    enhanced_real = cRM[..., 0] * noisy_real - cRM[..., 1] * noisy_imag
    enhanced_imag = cRM[..., 1] * noisy_real + cRM[..., 0] * noisy_imag
    predict = istft((enhanced_real, enhanced_imag), 512, 256, 512, length=noise.size(-1), input_type="real_imag").numpy()
    clean = clean.numpy()
    if data:
        noise = noise.numpy()
        return eval(clean, predict, text=text), predict, noise
    else:
        return eval(clean, predict, text=text)