import sys
sys.path.append('../') 
import argparse
import queue
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import librosa
import model
import torch
import scipy.signal as signal
import numpy as np
b, a = signal.butter(4, 100, 'highpass', fs=16000)
def init_net():
    device = 'cpu'
    checkpoint = torch.load('VibVoice_default.pt', map_location=torch.device(device))['model_state_dict']
    net = getattr(model, 'masker')('VibVoice_Early').to(device)
    net.eval() 
    net.load_state_dict(checkpoint, strict=True)
    return net
def inference_online(data, net):
    if data.shape[0] == 2:
        pass
    else:
        data = np.transpose(data)
    noisy = data[:1, :] * 4
    acc = data[1:, :] * 4

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
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=1000, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=100,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, default=320, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, default=16000, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
q = queue.Queue()
def callback(indata, frames, time, status):
    global previous_block
    indata = np.column_stack((indata[:, 1], indata[:, 0] * 4))
    # data = np.concatenate((previous_block, indata))
    # previous_block = indata
    # enhanced = inference_online(data, net)[args.blocksize:]
    # enhanced = enhanced[::args.downsample, np.newaxis]
    # print(enhanced.shape, indata.shape, enhanced.dtype, indata.dtype)
    # q_data = np.concatenate((indata[::args.downsample, :], indata[::args.downsample, :1]), axis=1)
    q_data = indata
    q.put(q_data)
def update_data(plot_lines):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata_input, plotdata_output, spec_input, spec_output
    shift_count = 0
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        shift_count += shift
        plotdata_input = np.roll(plotdata_input, -shift, axis=0)
        plotdata_input[-shift:, :] = data
    real_input = plotdata_input[-shift_count:, :]
    enhanced = inference_online(real_input, net)
    plotdata_output = np.roll(plotdata_output, -shift_count, axis=0)
    plotdata_output[-shift_count:, 0] = enhanced

    mel_in = librosa.feature.melspectrogram(y=real_input.T, sr=args.samplerate, n_mels=128, fmax=8000)
    mel_in = librosa.power_to_db(mel_in, ref=np.max)

    mel_out = librosa.feature.melspectrogram(y=enhanced, sr=args.samplerate, n_mels=128, fmax=8000)
    mel_out = librosa.power_to_db(mel_out, ref=np.max)
    shift = mel_in.shape[-1]
    spec_input = np.roll(spec_input, -shift, axis=2)
    spec_input[..., -shift:] = mel_in
    spec_output = np.roll(spec_output, -shift, axis=2)
    spec_output[0, :, -shift:] = mel_out

    for column, line in enumerate(lines1):
        line.set_ydata(plotdata_input[::args.downsample, 0])
    for column, line in enumerate(lines2):
        line.set_ydata(plotdata_input[::args.downsample, 1])
    for column, line in enumerate(lines3):
        line.set_ydata(plotdata_output[::args.downsample, 0])

    lines4.set_data(spec_input[0])
    lines5.set_data(spec_input[1])
    lines6.set_data(spec_output[0])

def init_wave_plot(title, ax):
    ax.set_title(title)
    ax.axis((0, len(plotdata_input)//args.downsample, -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)
net = init_net()
previous_block = np.zeros((args.blocksize, 2))
device = sd.default.device[0]
length = int(args.window * args.samplerate / (1000 * args.downsample))
length = int(args.window * args.samplerate / 1000)

plotdata_input = np.zeros((length, 2))
plotdata_output = np.zeros((length, 1))

spec_input = librosa.feature.melspectrogram(y=plotdata_input.T, sr=args.samplerate, n_mels=128, fmax=8000)
spec_input = librosa.power_to_db(spec_input, ref=np.max)
spec_output = librosa.feature.melspectrogram(y=plotdata_output.T, sr=args.samplerate, n_mels=128, fmax=8000)
spec_output = librosa.power_to_db(spec_output, ref=np.max)

fig, ax = plt.subplots(2, 3)
lines1 = ax[0, 0].plot(plotdata_input[::args.downsample, 0])
lines2 = ax[0, 1].plot(plotdata_input[::args.downsample, 1])
lines3 = ax[0, 2].plot(plotdata_output[::args.downsample, 0])

lines4 = ax[1, 0].imshow(spec_input[0], aspect='auto', origin='lower', vmin=-80, vmax=0)
lines5 = ax[1, 1].imshow(spec_input[1], aspect='auto', origin='lower', vmin=-80, vmax=0)
lines6 = ax[1, 2].imshow(spec_output[0], aspect='auto', origin='lower', vmin=-80, vmax=0)

init_wave_plot('Input_Audio', ax[0, 0])
init_wave_plot('Input_Vibration', ax[0, 1])
init_wave_plot('Output', ax[0, 2])
fig.tight_layout()

stream = sd.InputStream(channels=2, blocksize=args.blocksize,
    samplerate=args.samplerate, callback=callback)

ani = FuncAnimation(fig, update_data, interval=args.interval,)
try:
    with stream:
        plt.show()
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))
