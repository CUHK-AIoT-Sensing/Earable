import os
import json
import math
import time

import numpy as np
import torch
import torch.utils.data as Data
import scipy.signal as signal
import librosa
from feature import norm_amplitude, tailor_dB_FS, is_clipped
import soundfile as sf
import torchaudio as ta
import argparse
rate_mic = 16000
rate_imu = 1600
length = 3
stride = 3


sentences = [
    "HAPPY NEW YEAR PROFESSOR AUSTIN NICE TO MEET YOU",
    "WE WANT TO IMPROVE SPEECH QUALITY IN THIS PROJECT",
    "BUT WE DON'T HAVE ENOUGH DATA TO TRAIN OUR MODEL",
    "TRANSFER FUNCTION CAN BE A GOOD HELPER TO GENERATE DATA"
]

def snr_mix(noise_y, clean_y, snr, target_dB_FS, target_dB_FS_floating_value, rir=None, eps=1e-6):
        """
        Args:
            noise_y: 噪声
            clean_y: 纯净语音
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps
        Returns:
            (noisy_y，clean_y)
        """
        if rir is not None:
            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]
        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value
        )

        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar
        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar
        return noisy_y, clean_y

class NoiseDataset:
    def __init__(self, files=None, sample_rate=16000, silence_length=0.2, target_length=3, num_noises=1):
        """
        Special dataloader for Noise
        """
        self.files = files
        self.sr = sample_rate
        self.silence_length = silence_length
        self.target_length = target_length * self.sr
        self.num_noises = num_noises
    def __len__(self):
        return len(self.files)
    def __getitem__(self):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = self.target_length
        while remaining_length > 0:
            noise_file, info = self.files[np.random.randint(0, self.__len__())]
            noise_new_added, sr = sf.read(noise_file, dtype='float32')
            # noise_new_added, sr = ta.load(noise_file)
            # noise_new_added = noise_new_added[0]
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)
            # If still need to add new noise, insert a small silence segment firstly
            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len
        if len(noise_y) > self.target_length:
            idx_start = np.random.randint(len(noise_y) - self.target_length)
            noise_y = noise_y[idx_start : idx_start + self.target_length]
        return noise_y
class BaseDataset:
    def __init__(self, files=None, pad=False, sample_rate=16000):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.sample_rate = sample_rate
        self.length = length
        self.stride = stride
        for info in self.files:
            _, file_length = info
            if self.length is None:
                examples = 1
            elif file_length < self.length*self.sample_rate:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length * self.sample_rate) / self.stride/self.sample_rate) + 1)
            else:
                examples = (file_length - self.length * self.sample_rate) // (self.stride*self.sample_rate) + 1
            self.num_examples.append(examples)
    def __len__(self):
        return sum(self.num_examples)
    def __getitem__(self, index):
        for info, examples in zip(self.files, self.num_examples):
            file, _ = info
            if index >= examples:
                index -= examples
                continue
            duration = 0
            offset = 0
            if self.length:
                offset = self.stride * index
                duration = self.length
            if file[-3:] == 'txt':
                data = np.loadtxt(file)
                data = data[offset * self.sample_rate: (offset + duration) * self.sample_rate, :]
                data /= 2 ** 14
                b, a = signal.butter(4, 80, 'highpass', fs=self.sample_rate)
                data = signal.filtfilt(b, a, data, axis=0)
                data = np.clip(data, -0.05, 0.05)
            else:
                data, sr = ta.load(file, frame_offset=offset * self.sample_rate, num_frames=duration * self.sample_rate)
                data = data[0].numpy()
            return data, file
class NoisyCleanSet:
    def __init__(self, json_paths, text=False, person=None, simulation=False, ratio=1, snr=(0, 20),
                 rir=None):
        '''
        :param json_paths: speech (clean), noisy/ added noise, IMU (optional)
        :param text: whether output the text, only apply to Sentences
        :param person: person we want to involve
        :param simulation: whether the noise is simulation
        :param time_domain: use frequency domain (complex) or time domain
        :param ratio: ratio of the data we use
        :param snr: SNR range of the synthetic dataset
        '''
        self.dataset = []
        self.ratio = ratio
        self.simulation = simulation
        self.text = text
        self.snr_list = np.arange(snr[0], snr[1], 1)
        if len(json_paths) == 2:
            # only clean + noise
            self.augmentation = True
        else:
            self.augmentation = False
        sr = [16000, 16000, 1600]
        for i, path in enumerate(json_paths):
            with open(path, 'r') as f:
                data = json.load(f)
            if person is not None and isinstance(data, dict):
                tmp = []
                for p in person:
                    if ratio > 0:
                        tmp += data[p][:int(len(data[p]) * self.ratio)]
                    else:
                        tmp += data[p][int(len(data[p]) * self.ratio):]
                data = tmp
            else:
                if ratio > 0:
                    data = data[:int(len(data) * self.ratio)]
                else:
                    data = data[int(len(data) * self.ratio):]
            if self.simulation and i == 1:
                self.dataset.append(NoiseDataset(data, sample_rate=sr[i], target_length=length))
            else:
                self.dataset.append(BaseDataset(data, sample_rate=sr[i]))
        self.noise_length = len(self.dataset[1])
        self.rir = rir
        if self.rir is not None:
            with open(rir, 'r') as f:
                data = json.load(f)
            self.rir = data
            self.rir_length = len(self.rir)
    def __getitem__(self, index):
        clean, file = self.dataset[0][index]
        if self.simulation:
            # use rir dataset to add noise
            use_reverb = False if self.rir is None else bool(np.random.random(1) < 0.75)
            noise = self.dataset[1].__getitem__()
            random_snr = np.random.choice(self.snr_list)
            noise, clean = snr_mix(noise, clean, random_snr, -25, 10,
            rir = sf.read(self.rir[np.random.randint(0, self.rir_length)][0], dtype='float32')[0] if use_reverb else None, eps=1e-6)
        else:
            noise, _ = self.dataset[1][index]
        data = [clean.astype(np.float32), noise.astype(np.float32)]
        if not self.augmentation:
            acc, _ = self.dataset[2][index]
            acc = np.transpose(acc)
            data.append(acc.astype(np.float32))
        if self.text:
            sentence = sentences[int(file.split('/')[4][-1])-1]
            data.append(sentence)
        return data
    def __len__(self):
        return len(self.dataset[0])
class EMSBDataset:
    def __init__(self, json_paths, text=False, person=None, simulation=False, time_domain=False,
                 ratio=1, snr=(0, 20), rir='json/rir_noise.json'):
        '''
        :param json_paths: speech (clean), noisy/ added noise, IMU (optional)
        :param text: whether output the text, only apply to Sentences
        :param person: person we want to involve
        :param simulation: whether the noise is simulation
        :param time_domain: use frequency domain (complex) or time domain
        :param ratio: ratio of the data we use
        :param snr: SNR range of the synthetic dataset
        '''
        self.dataset = []
        self.ratio = ratio
        self.simulation = simulation
        self.text = text
        self.time_domain = time_domain
        self.snr_list = np.arange(snr[0], snr[1], 1)
        sr = 16000
        for i, path in enumerate(json_paths):
            with open(path, 'r') as f:
                data = json.load(f)
                if person is not None and isinstance(data, dict):
                    datasets = []
                    for p in person:
                        dataset = BaseDataset(data[p], sample_rate=sr)
                        size1 = int(len(dataset) * self.ratio)
                        size2 = len(dataset) - size1
                        dataset, _ = torch.utils.data.random_split(dataset, [size1, size2])
                        datasets.append(dataset)
                    dataset = torch.utils.data.ConcatDataset(datasets)
                else:
                    if ratio > 0:
                        data = data[:int(len(data) * self.ratio)]
                    else:
                        data = data[int(len(data) * self.ratio):]
                    dataset = BaseDataset(data, sample_rate=sr)
            self.dataset.append(dataset)
        self.rir = rir
        if self.rir is not None:
            with open(rir, 'r') as f:
                data = json.load(f)
            self.rir = data
            self.rir_length = len(self.rir)
        self.noise_length = len(self.dataset[1])
    def __getitem__(self, index):
        data, file = self.dataset[0][index]
        clean = data[0]
        imu = data[1, ::10]
        if self.simulation:
            # use rir dataset to add noise
            use_reverb = False if self.rir is None else bool(np.random.random(1) < 0.75)
            noise, _ = self.dataset[1][np.random.randint(0, self.noise_length)]
            snr = np.random.choice(self.snr_list)
            noise, clean = snr_mix(noise, clean, snr, -25, 10,
            rir = librosa.load(self.rir[np.random.randint(0, self.rir_length)][0], sr=rate_mic, mono=False)[0]
            if use_reverb else None, eps=1e-6)
        else:
            noise, _ = self.dataset[1][index]
        if self.text:
            setence = sentences[int(file.split('/')[4][-1])-1]
            return setence, imu, noise, clean
        else:
            return imu, noise, clean
    def __len__(self):
        return len(self.dataset[0])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-pre train, 1-main benchmark, 2-mirco benchmark')
    args = parser.parse_args()
    if args.mode == 0:
        # check data
        dataset_train = NoisyCleanSet(['json/train.json', 'json/dev.json'], simulation=True, ratio=1)
        loader = Data.DataLoader(dataset=dataset_train, batch_size=2, shuffle=False)
        for step, (clean, noise) in enumerate(loader):
            print(noise.shape, clean.shape)