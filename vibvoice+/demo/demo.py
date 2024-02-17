import model
import torch
import scipy.signal as signal
import numpy as np

def init_net():
    device = 'cpu'
    checkpoint = torch.load('VibVoice_default.pt', map_location=torch.device(device))['model_state_dict']
    net = getattr(model, 'masker')('VibVoice_Early').to(device)
    net.eval() 
    net.load_state_dict(checkpoint, strict=True)
    return net
def inference_online(data, net):
    data = np.transpose(data)
    noisy = data[:1, :] * 4
    acc = data[1:, :] * 4

    b, a = signal.butter(4, 100, 'highpass', fs=16000)
    noisy = signal.filtfilt(b, a, noisy)
    acc = signal.filtfilt(b, a, acc)

    noisy = torch.from_numpy(noisy.copy()).to(dtype=torch.float)
    acc = torch.from_numpy(acc.copy()).to(dtype=torch.float)
    noisy_stft = torch.stft(noisy, 640, 320, 640, window=torch.hann_window(640, device=noisy.device), return_complex=True, center=False).unsqueeze(0)
    noisy_mag, noisy_phase = torch.abs(noisy_stft), torch.angle(noisy_stft)
    acc_stft = torch.stft(acc, 640, 320, 640, window=torch.hann_window(640, device=acc.device), return_complex=True, center=False).unsqueeze(0)
    acc_mag = torch.abs(acc_stft)
    length = noisy_mag.shape[-1]
    real_time_output = []
    with torch.no_grad():
        for i in range(length):
            real_time_input1 = noisy_mag[:, :, :, i:i+1]
            real_time_input2 = acc_mag[:, :, :, i:i+1]
            real_time_output.append(net.forward_causal(real_time_input1, real_time_input2))
    real_time_output = torch.cat(real_time_output, dim=-1)
    features = torch.complex(real_time_output * torch.cos(noisy_phase), real_time_output * torch.sin(noisy_phase))
    est_audio = torch.istft(features[0,0], 640, 320, 640, window=torch.hamming_window(642, device=features.device)[1:-1], length=None, center=False).cpu().numpy()
    return est_audio


