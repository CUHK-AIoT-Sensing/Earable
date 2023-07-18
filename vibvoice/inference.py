import numpy as np
import scipy.signal as signal
import torchaudio as ta
import os
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

'''
pipeline:
: please collect data first, the data can be un-synchronized
1) resample imu data to 1600
2) synchronize imu and audio
3) inference
'''
rate_mic = 16000
rate_imu = 1600
length = 5
stride = 5

def synchronization(audio, imu):
    time_imu = float(imu.split('_')[1][:-4])
    time_mic = float(audio.split('_')[1][:-4])
    shift = int((time_imu - time_mic) * 1600)
    return shift
def resample(data, shift):
    timestamp = data[:, -1]
    data = data[:, :3]
    print(timestamp.shape, timestamp[-1] - timestamp[0])
    f = interpolate.interp1d(timestamp - timestamp[0], data, axis=0, kind='linear')
    t = min((timestamp[-1] - timestamp[0]), length)
    num_sample = int(length * rate_imu)
    data = np.zeros((num_sample, 3))
    xnew = np.linspace(0, t, num_sample)
    data[shift:num_sample, :] = f(xnew)[:-shift, :]
    return data
def preprocess(audio, imu):
    imu = np.loadtxt(imu)
    imu /= 2 ** 14
    b, a = signal.butter(4, 80, 'highpass', fs=1600)
    imu = signal.filtfilt(b, a, imu, axis=0)
    imu = np.clip(imu, -0.05, 0.05)

    audio, _ = ta.load(audio)
    audio = audio[0].numpy()
    return audio, imu
if __name__ == "__main__":
    folder = 'examples'
    files = os.listdir(folder)
    index = 0
    imu_file = files[index]
    mic_file = files[index + 10]
    airpods_file = files[index + 15]
    shift = synchronization(mic_file, imu_file)
    print(shift)
    audio, imu = preprocess(os.path.join(folder, mic_file), os.path.join(folder, imu_file))
    fig, axs= plt.subplots(4, 1)
    axs[0].plot(imu[:, 0])
    axs[1].plot(imu[:, 1])
    axs[2].plot(imu[:, 2])
    axs[3].plot(audio)
    plt.savefig('examples/plot.png')
    