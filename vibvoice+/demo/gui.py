import sys
sys.path.append('../') 
import model
import torch
import time
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import PySimpleGUI as sg

def baseline_inference(before, fs):
    audio = before[0, :]
    vib = before[1, :]
    b, a  = signal.butter(4, 2000, 'highpass', fs=fs)
    filtered_audio = signal.filtfilt(b, a, audio)
    baseline = filtered_audio + vib
    return baseline 
def plot(fname):
    fs, before = wavfile.read(audio_folder + fname)
    before, fs = librosa.load(audio_folder + fname, mono=False, sr=None)
    audio = before[0, :]
    S_beofe = librosa.feature.melspectrogram(y=audio, sr=fs)

    # fs, after = wavfile.read(audio_folder + 'enhanced_' + fname)
    after, fs = librosa.load(audio_folder + 'enhanced_' + fname, sr=None)
    S_after = librosa.feature.melspectrogram(y=after, sr=fs)

    # fs, baseline = wavfile.read(audio_folder + 'baseline_' + fname)
    baseline, fs = librosa.load(audio_folder + 'baseline_' + fname, sr=None)
    S_baseline = librosa.feature.melspectrogram(y=baseline, sr=fs)
    plt.get_current_fig_manager().resize(1440, 720)

    plt.subplot(2, 3, 1)
    plt.plot(audio)
    plt.title('Audio w/o processing')
    plt.subplot(2, 3, 4)
    librosa.display.specshow(librosa.power_to_db(S_beofe, ref=np.max), y_axis='mel', sr=fs, x_axis='time')

    plt.subplot(2, 3, 2)
    plt.plot(baseline)
    plt.title('Baseline output')
    plt.subplot(2, 3, 5)
    librosa.display.specshow(librosa.power_to_db(S_baseline, ref=np.max), y_axis='mel', sr=fs, x_axis='time')

    plt.subplot(2, 3, 3)
    plt.plot(after)
    plt.title('VibVoice output')
    plt.subplot(2, 3, 6)
    librosa.display.specshow(librosa.power_to_db(S_after, ref=np.max), y_axis='mel', sr=fs, x_axis='time')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.2, hspace=0.1)
    plt.show(block=False)

def inference_file(data_name = 'example1.wav', plot_flag=True):
    device = 'cpu'
    checkpoint = torch.load('VibVoice_default.pt', map_location=torch.device(device))['model_state_dict']
    net = getattr(model, 'masker')('VibVoice_Early').to(device)
    net.eval() 
    net.load_state_dict(checkpoint, strict=True)

    # _, data = wavfile.read(audio_folder + data_name)
    data, sample_rate = librosa.load(audio_folder + data_name, mono=False, sr=None)
    # data = np.transpose(data)
    noisy = data[:1, :] * 4
    acc = data[1:, :] * 4

    b, a = signal.butter(4, 100, 'highpass', fs=16000)
    noisy = signal.filtfilt(b, a, noisy)
    acc = signal.filtfilt(b, a, acc)

    noisy = torch.from_numpy(noisy.copy()).to(dtype=torch.float)
    acc = torch.from_numpy(acc.copy()).to(dtype=torch.float)

    complex_stft = torch.stft(noisy, 640, 320, 640, window=torch.hann_window(640, device=noisy.device), return_complex=True).unsqueeze(0)
    noisy_mag, noisy_phase = torch.abs(complex_stft), torch.angle(complex_stft)
    complex_stft = torch.stft(acc, 640, 320, 640, window=torch.hann_window(640, device=acc.device), return_complex=True).unsqueeze(0)
    acc_mag = torch.abs(complex_stft)
        
    length = noisy_mag.shape[-1]

    real_time_output = []
    t_start = time.time()
    with torch.no_grad():
        for i in range(length):
            real_time_input1 = noisy_mag[:, :, :, i:i+1]
            real_time_input2 = acc_mag[:, :, :, i:i+1]
            real_time_output.append(net.forward_causal(real_time_input1, real_time_input2))
        full_output = net(noisy_mag, acc_mag)
    real_time_output = torch.cat(real_time_output, dim=-1)
    t_end = time.time()

    # measure whether real-time is different from full output
    error = torch.mean(torch.abs((full_output - real_time_output)/(full_output +1e-6)))
    print('Real-time difference ', error.item())

    # summary all the latency
    latency = (t_end - t_start)
    inference_latency_per_frame = round(latency/length ,4)
    load_latency_per_frame = 1/50
    overall_latency_per_frame = inference_latency_per_frame + load_latency_per_frame
    print('Latency:', round(overall_latency_per_frame,2), 'RTF:', round(inference_latency_per_frame/load_latency_per_frame, 2))

    real_time_output = real_time_output.squeeze()
    noisy_phase = noisy_phase.squeeze()
    features = torch.complex(real_time_output * torch.cos(noisy_phase), real_time_output * torch.sin(noisy_phase))
    est_audio = torch.istft(features, 640, 320, 640, window=torch.hann_window(640, device=features.device), length=None).cpu().numpy()
    wavfile.write(audio_folder + 'enhanced_' + data_name, 16000, est_audio)

    wavfile.write(audio_folder + 'baseline_' + data_name, 16000, baseline_inference(data, 16000))

    # save as wav
    if plot_flag:
        plot(data_name)
    return round(latency, 2), round(latency/(length/50), 2)
def tailor_dB_FS(y, target_dB_FS=-15, eps=1e-6):
    rms = np.mean(y ** 2)**0.5
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y
sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('This is the Demo GUI for VibVoice, 1) Select ongoing noise (if any), 2) Start online recording (not neccessary to play noise), 3) Use the pre-recorded audio, 4) Plot the results, 5) Play the results', size=(40, 4), font=('Helvetica', 20))],
            [sg.Text('Not Started', key='content', size=(50, 2), font=('Helvetica', 20))],
            [sg.Listbox(values=['car.wav', 'S0724W0121.wav', 'S0725W0121.wav', 'S0726W0121.wav'], key='noise', size=(20, 4)), sg.Button('Select noise', size=20)],
            [sg.Button('Start recording', size=20)],

            [sg.Text('English example', size=(20, 1)), sg.Text('Mandarin example', size=(20, 1))],
            [sg.Listbox(values=['en_example1.wav', 'en_example2.wav', 'en_example3.wav'], key='audio1', size=(20, 4)), 
             sg.Listbox(values=['cn_example1.wav', 'cn_example2.wav', 'cn_example3.wav'], key='audio2', size=(20, 4)),
             sg.Button('Select audio', size=20)],
            [sg.Button('Inference', size=20)],
            [sg.Button('Plot', size=20)],
            [sg.Button('Play input', size=20), sg.Button('Play baseline', size=20), sg.Button('Play Vibvoice', size=20)],
            [sg.Button('exit', size=20)]
            ]
# Create the Window
window = sg.Window('VibVoice Demo', layout)
audio_folder = 'audio/'
noise_folder = 'noise/'
fname = None
#fname = 'online_recording.wav'
playnoise = None
while True:
    event, values = window.read()
    if event == 'Start recording':
        if playnoise is not None:
            fs, noise = wavfile.read(noise_folder + playnoise)
            if len(noise) > fs * 5:
                start_index = np.random.randint(0, len(noise) - fs * 5)
                noise = noise[start_index : start_index + fs*5]
                noise = noise / np.max(np.abs(noise))
            myrecording = sd.playrec(noise, samplerate=16000, channels=2, blocking=True)
            sd.wait()
        else:
            myrecording = sd.rec(16000*5, samplerate=16000, channels=2, blocking=True)
            sd.wait()
        vibration, audio = myrecording[:, 0], myrecording[:, 1]
        vibration *= 4 # amplify the vibration
        data = np.stack((audio, vibration)).T
        fname = 'online_recording.wav'
        window['content'].update("finish recording {}".format(fname))
        wavfile.write(audio_folder + fname, 16000, data)
    elif event == 'Select noise':
        if len(values['noise']) == 0:
            sg.popup("No item selected!", title="Warning")
        else:
            playnoise = values['noise'][0]
            window['content'].update("select noise {}".format(playnoise))
    elif event == 'Select audio':
        if len(values['audio1']) == 0 and len(values['audio2']) == 0:
            sg.popup("No item selected!", title="Warning")
        elif len(values['audio1']) != 0 and len(values['audio2']) != 0:
            sg.popup("Multiple items selected!", title="Warning")
            for key in ['audio1', 'audio2']:
                window[key].update(set_to_index=[])
        else:
            for key in ['audio1', 'audio2']:
                if len(values[key]) != 0:
                    fname = values[key][0]
                window['content'].update("select audio {}".format(fname))
    elif event == 'Inference':
        if fname is None:
            sg.popup("No file selected!", title="Warning")
        else:
            latency, RTF = inference_file(fname, plot_flag=False)
            window['content'].update('Inference done, latency: {}, Real-time factor: {}'.format(latency, RTF))
    elif event == 'Plot':
        if fname is None:
            sg.popup("No file selected!", title="Warning")
        else:
            plot(fname)
    elif event == 'Play input':
        audio, fs = librosa.load(audio_folder + fname, sr=None, mono=False)
        audio = tailor_dB_FS(audio[0])
        sd.play(audio, fs)
        sd.wait()
    elif event == 'Play baseline':
        audio, fs = librosa.load(audio_folder + fname, sr=None)
        audio = tailor_dB_FS(audio)
        sd.play(audio, fs)
        sd.wait()    
    elif event == 'Play Vibvoice':
        audio, fs = librosa.load(audio_folder + fname, sr=None)
        audio = tailor_dB_FS(audio)
        sd.play(audio, fs)
        sd.wait()   
    elif event == sg.WIN_CLOSED or event == 'exit': # if user closes window or clicks cancel
        break
window.close()
