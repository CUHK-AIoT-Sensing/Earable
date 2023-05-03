import librosa
import os
import numpy as np
import argparse
import scipy.signal as signal
from skimage import filters
import shutil
seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32
rate_mic = 16000
rate_imu = 1600
T = 30
segment = 5
stride = 5
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(segment * rate_mic/(seg_len_mic-overlap_mic)) + 1
time_stride = int(stride * rate_mic/(seg_len_mic-overlap_mic))
def normalization(wave_data, rate=16000, T=5):
    b, a = signal.butter(4, 100, 'highpass', fs=rate)
    wave_data = signal.filtfilt(b, a, wave_data)
    if len(wave_data) >= T * rate:
        return wave_data[:T * rate]
    wave_data = np.pad(wave_data, (0, T * rate - len(wave_data)), 'constant', constant_values=(0, 0))
    return wave_data
def frequencydomain(wave_data, seg_len=2560, overlap=2240, rate=16000, mfcc=False):
    if mfcc:
        Zxx = librosa.feature.melspectrogram(wave_data, sr=rate, n_fft=seg_len, hop_length=seg_len-overlap, power=1)
        return Zxx, None
    else:
        f, t, Zxx = signal.stft(wave_data, nperseg=seg_len, noverlap=overlap, fs=rate)
        phase = np.exp(1j * np.angle(Zxx))
        Zxx = np.abs(Zxx)
        return Zxx, phase
def load_audio(name, T, seg_len=2560, overlap=2240, rate=16000, normalize=False, mfcc=False):
    wave, _ = librosa.load(name, sr=rate)
    if normalize:
        wave = normalization(wave, rate, T)
    Zxx, phase = frequencydomain(wave, seg_len=seg_len, overlap=overlap, rate=rate, mfcc=mfcc)
    return wave, Zxx, phase

def read_data(file, seg_len=256, overlap=224, rate=1600, mfcc=False, filter=True):
    fileobject = open(file, 'r')
    lines = fileobject.readlines()
    data = np.zeros((len(lines), 4))
    for i in range(len(lines)):
        line = lines[i].split(' ')
        data[i, :] = [float(item) for item in line]
    data[:, :-1] /= 2**14
    if filter:
        b, a = signal.butter(4, 10, 'highpass', fs=rate)
        data[:, :3] = signal.filtfilt(b, a, data[:, :3], axis=0)
        data[:, :3] = np.clip(data[:, :3], -0.05, 0.05)
    if mfcc:
        Zxx = []
        for i in range(3):
            Zxx.append(librosa.feature.melspectrogram(data[:, i], sr=rate, n_fft=seg_len, hop_length=seg_len-overlap, power=1))
        Zxx = np.array(Zxx)
        Zxx = np.linalg.norm(Zxx, axis=0)
    else:
        Zxx = signal.stft(data[:, :3], nperseg=seg_len, noverlap=overlap, fs=rate, axis=0)[-1]
        Zxx = np.linalg.norm(np.abs(Zxx), axis=1)
    return data, Zxx
def synchronization(Zxx, imu):
    in1 = np.sum(Zxx[:freq_bin_high, :], axis=0)
    in2 = np.sum(imu, axis=0)
    shift = np.argmax(signal.correlate(in1, in2)) - len(in2)
    return np.roll(imu, shift, axis=1)
def estimate_response(imu, Zxx):
    select1 = Zxx > 1 * filters.threshold_otsu(Zxx)
    select2 = imu > 1 * filters.threshold_otsu(imu)
    select = select2 & select1
    Zxx_ratio = np.divide(imu, Zxx, out=np.zeros_like(imu), where=select)
    response = np.zeros((2, freq_bin_high))
    for i in range(freq_bin_high):
        if np.sum(select[i, :]) > 0:
            response[0, i] = np.mean(Zxx_ratio[i, :], where=select[i, :])
            response[1, i] = np.std(Zxx_ratio[i, :], where=select[i, :])
    return response
def transfer_function(clip1, clip2, response):
    new_response = estimate_response(clip1, clip2)
    response = 0.25 * new_response + 0.75 * response
    return response

def filter_function(response):
    m = np.max(response)
    n1 = np.mean(response[-5:])
    n2 = np.mean(response)
    if m > 35:
        return False
    elif (2*n1) > m:
        return False
    else:
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', action="store", type=str, default='', required=False)
    parser.add_argument('--two_channel', action="store", type=str, default='', required=False)
    args = parser.parse_args()
    candidate = os.listdir(args.data_dir)
    if not os.path.exists('transfer_function/'):
        os.makedirs('transfer_function/')
    else:
        shutil.rmtree('transfer_function/')
        os.makedirs('transfer_function/')
    for i in range(len(candidate)):
        print('processing number {}'.format(i))
        Zxx_valid = [[]] * freq_bin_high
        name = candidate[i]
        count = 0
        error = []
        path = os.path.join(args.data_dir, name, 'clean')
        files = os.listdir(path)
        files.sort()
        N = int(len(files) / 4)
        files_imu1 = files[:N]
        files_imu2 = files[N : 2 * N]
        files_mic1 = files[2 * N:3 * N]
        files_mic2 = files[3 * N:]

        for index in range(N):
            response = np.zeros((2, freq_bin_high))
            wave, Zxx, phase = load_audio(path + '/' + files_mic1[index], T, seg_len_mic, overlap_mic, rate_mic, normalize=True)
            data, imu = read_data(path + '/' + files_imu1[index], seg_len_imu, overlap_imu, rate_imu)
            imu = synchronization(Zxx, imu)
            for j in range(int((T - segment) / stride) + 1):
                clip2 = Zxx[:freq_bin_high, j * time_stride:j * time_stride + time_bin]
                clip1 = imu[:, j * time_stride:j * time_stride + time_bin]
                response = transfer_function(clip1, clip2, response)

            if filter_function(response):
                np.savez('transfer_function/' + str(i) + '_' + str(count) + '.npz', response=response[0, :], variance=response[1, :])
                count += 1
        if args.two_channel:
            for index in range(N):
                response = np.zeros((2, freq_bin_high))
                wave, Zxx, phase = load_audio(path + '/' + files_mic1[index], T, seg_len_mic, overlap_mic, rate_mic, normalize=True)
                data, imu = read_data(path + '/' + files_imu2[index], seg_len_imu, overlap_imu, rate_imu)
                imu = synchronization(Zxx, imu)
                for j in range(int((T - segment) / stride) + 1):
                    clip2 = Zxx[:freq_bin_high, j * time_stride:j * time_stride + time_bin]
                    clip1 = imu[:, j * time_stride:j * time_stride + time_bin]
                    response = transfer_function(clip1, clip2, response)

                if filter_function(response):
                    np.savez('transfer_function/' + str(i) + '_' + str(count) + '.npz', response=response[0, :], variance=response[1, :])
                    count += 1
