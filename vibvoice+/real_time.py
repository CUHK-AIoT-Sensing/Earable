from model import DPCRN
import torch
import time
import torchaudio
import scipy.signal as signal
from feature import stft, istft
import scipy.io.wavfile as wavfile
from dataset import vad_annotation
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action="store", default=None, type=str, required=False)
    parser.add_argument('--stream', action="store_true", default=False, required=False)
    args = parser.parse_args()

    device = 'cpu'
    checkpoint = torch.load('checkpoints/20230918-190354/best.pth')
    net = DPCRN().to(device)
    net.eval() 
    net.load_state_dict(checkpoint, strict=True)

    if args.data is not None:
        data, _ = torchaudio.load(args.data)
        noisy = data[1:, :] * 4
        acc = data[:1, :] * 2 * 4

        b, a = signal.butter(4, 100, 'highpass', fs=16000)
        b = torch.from_numpy(b, ).to(dtype=torch.float)
        a = torch.from_numpy(a, ).to(dtype=torch.float)
        x = torchaudio.functional.filtfilt(noisy, a, b,)
        acc = torchaudio.functional.filtfilt(acc, a, b,) 

        vad = vad_annotation(acc)

        noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy.unsqueeze(1).to(device), 640, 320, 640)
        acc, _, _, _ = stft(acc.unsqueeze(1).to(device), 640, 320, 640)

    else:
        noisy_mag = torch.randn(1, 1, 321, 251).to(device)
        acc = torch.randn(1, 1, 321, 251).to(device)
        vad = 1
    length = noisy_mag.shape[-1]

    real_time_output = []
    t_start = time.time()
    with torch.no_grad():
        for i in range(length):
            real_time_input1 = noisy_mag[:, :, :, i:i+1]
            real_time_input2 = acc[:, :, :, i:i+1]
            real_time_output.append(net.forward_causal(real_time_input1, real_time_input2))
        full_output = net(noisy_mag, acc)
    real_time_output = torch.cat(real_time_output, dim=-1)
    t_end = time.time()

    full_output = full_output * vad.reshape(1,1,1,-1)
    real_time_output = real_time_output * vad.reshape(1,1,1,-1)

    # measure whether real-time is different from full output
    error = torch.mean(torch.abs((full_output - real_time_output)/(full_output +1e-6)))
    print('Real-time difference ', error.item())

    # summary all the latency
    inference_latency_per_frame = round((t_end - t_start)/length ,4)
    load_latency_per_frame = 1/50
    overall_latency_per_frame = inference_latency_per_frame + load_latency_per_frame
    print('Latency:', overall_latency_per_frame, 'RTF:', inference_latency_per_frame/load_latency_per_frame)

    # save as wav
    if args.data:
        est_audio = istft((real_time_output.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")    
        est_audio = est_audio.cpu().numpy()
        wavfile.write('enhanced_' + args.data, 16000, est_audio[0])
