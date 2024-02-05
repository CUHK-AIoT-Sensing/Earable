import json
import math
import numpy as np
import scipy.signal as signal
import torchaudio
import soundfile as sf
import torch
from feature import tailor_dB_FS, snr_mix, d_vector
from resemblyzer import VoiceEncoder
from tqdm import tqdm
'''
https://github.com/resemble-ai/Resemblyzer
'''
def vad_annotation(audio):
    '''
    according to "In-Ear-Voice: Towards Milli-Watt Audio Enhancement With Bone-Conduction Microphones for In-Ear Sensing Platforms, IoTDI'23"
    '''
    vad = torch.zeros((1, audio.shape[-1]//320+1), dtype=torch.float)
    spec = (torch.abs(torch.stft(audio, 640, 320, 640, return_complex=True)))
    spec = spec.sum(dim=1) ** 0.5
    spec = torch.nn.functional.avg_pool1d(spec, 20, 1, padding=10)[:, 1:]
    threshold = spec.min() + 1 * spec.mean()
    vad[spec > threshold] = 1
    return vad

class BaseDataset:
    def __init__(self, files=None, pad=True, sample_rate=16000, length=5, stride=3):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.sample_rate = sample_rate
        self.length = length
        self.stride = stride
        self.pad = True
        for info in files:
            _, file_length = info
            if self.length is None:
                examples = 1
            elif file_length < (self.length*self.sample_rate):
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
            file, samples = info
            if index >= examples:
                index -= examples
                continue
            if self.length is None:
                # data, _ = torchaudio.load(file)
                data, _ = sf.read(file, always_2d=True, dtype='float32')
                data = torch.from_numpy(data).permute(1, 0)
                return data, file
            else:
                # data, _ = torchaudio.load(file, frame_offset=self.stride * index * self.sample_rate, num_frames=self.length * self.sample_rate)
                data, _ = sf.read(file, frames=self.length * self.sample_rate, start=self.stride * index * self.sample_rate, always_2d=True, dtype='float32')
                data = torch.from_numpy(data).permute(1, 0)
                if data.shape[-1] < (self.sample_rate * self.length):
                    pad_before = np.random.randint((self.sample_rate *self.length) - data.shape[-1])
                    pad_after = (self.sample_rate *self.length) - data.shape[-1] - pad_before
                    data = torch.nn.functional.pad(data, (pad_before, pad_after, 0, 0), 'constant')
                return data, file
class VoiceBankDataset():
     def __init__(self, data, noise=None, snr=(-5, 15), rir=None, mode = 'PN'):
        self.snr_list = np.arange(snr[0], snr[1], 1)
        sr = 16000
        with open(data, 'r') as f:
            data = json.load(f)
            self.left_dataset = BaseDataset(data, sample_rate=sr)
        self.noise = noise
        if self.noise is not None:
            self.noise_dataset = BaseDataset(noise, sample_rate=sr)
            self.noise_length = len(self.noise_dataset)
        self.rir = rir
        if self.rir is not None:
            with open(rir, 'r') as f:
                data = json.load(f)
            self.rir = data
            self.rir_length = len(self.rir)
        self.b, self.a = signal.butter(4, 100, 'highpass', fs=16000)
        self.b = torch.from_numpy(self.b, ).to(dtype=torch.float)
        self.a = torch.from_numpy(self.a, ).to(dtype=torch.float)
        self.mode = mode
     def __len__(self):
        return len(self.left_dataset)
     def __getitem__(self, index):
        left, file = self.left_dataset[index]
        clean = left[:1, :]
        imu = left[1:, :]
        noise, _ = self.noise_dataset.__getitem__(np.random.randint(0, self.noise_length))

        if noise.shape[-1] <= clean.shape[-1]:
            noise = torch.nn.functional.pad(noise, (0, clean.shape[-1] - noise.shape[-1], 0, 0), 'constant')
        else:
            offset = np.random.randint(0, noise.shape[-1] - clean.shape[-1])
            noise = noise[:, offset:offset+clean.shape[-1]]

        use_reverb = False if self.rir is None else bool(np.random.random(1) < 0.75)
        rir = torchaudio.load(self.rir[np.random.randint(0, self.rir_length)][0])[0] if use_reverb else None
        snr = np.random.choice(self.snr_list)
        
        clean = torchaudio.functional.filtfilt(clean, self.a, self.b,)
        imu = torchaudio.functional.filtfilt(imu, self.a, self.b,)

        target_dB_FS = np.random.randint(-35, -15)
        noisy, clean, noise, scaler = snr_mix(noise, clean, snr, target_dB_FS, rir, eps=1e-6)
        imu = tailor_dB_FS(imu, target_dB_FS)[0]

        mixture = torch.cat([clean, noise], dim=0)
        return {'imu': imu, 'clean': clean, 'vad': vad_annotation(clean), 'noisy': noisy, 'file': file, 'noise': noise, 'mixture': mixture}  
class ABCSDataset():
    def __init__(self, data_json, noise=None, snr=(-5, 15), rir=None, length=5):
        self.snr_list = np.arange(snr[0], snr[1], 1)
        sr = 16000
        with open(data_json, 'r') as f:
            json_data = json.load(f)
        data = []
        for speaker in json_data.keys():
            data += json_data[speaker]
        self.left_dataset = BaseDataset(data, sample_rate=sr, length=length)
        self.prepare_vector_offline(data_json, data)

        self.noise = noise
        if self.noise is not None:
            self.noise_dataset = BaseDataset(noise, sample_rate=sr)
            self.noise_length = len(self.noise_dataset)
        self.rir = rir
        if self.rir is not None:
            with open(rir, 'r') as f:
                data = json.load(f)
            self.rir = data
            self.rir_length = len(self.rir)
        self.b, self.a = signal.butter(4, 100, 'highpass', fs=16000)
        self.b = torch.from_numpy(self.b, ).to(dtype=torch.float)
        self.a = torch.from_numpy(self.a, ).to(dtype=torch.float)

    def __len__(self):
        return len(self.left_dataset)
    def prepare_vector_offline(self, data_json, data):
        embedder = VoiceEncoder('cuda')
        save_name = data_json.replace('.json', '.npy')
        import os
        if os.path.exists(save_name):
            self.vectors = np.load(save_name, allow_pickle=True).item()
        else:
            self.vectors = {}
            count = {}
            for (file, samples) in tqdm(data):
                person = os.path.basename(file).split('_')[0]
                vector = d_vector(file, embedder)
                self.vectors[file] = vector
                if person not in self.vectors:
                    self.vectors[person] = vector
                    count[person] = 1
                else:
                    self.vectors[person] += vector
                    count[person] += 1
            for person in count.keys():
                self.vectors[person] /= count[person]
            np.save(save_name, self.vectors)
    def prepare_vector_online(self,):
        return
    def __getitem__(self, index):
        left, file = self.left_dataset[index]
        clean = left[:1, :]
        imu = left[1:, :]
        noise, _ = self.noise_dataset.__getitem__(np.random.randint(0, self.noise_length))

        if noise.shape[-1] <= clean.shape[-1]:
            noise = torch.nn.functional.pad(noise, (0, clean.shape[-1] - noise.shape[-1], 0, 0), 'constant')
        else:
            offset = np.random.randint(0, noise.shape[-1] - clean.shape[-1])
            noise = noise[:, offset:offset+clean.shape[-1]]

        use_reverb = False if self.rir is None else bool(np.random.random(1) < 0.75)
        rir = torchaudio.load(self.rir[np.random.randint(0, self.rir_length)][0])[0] if use_reverb else None
        snr = np.random.choice(self.snr_list)
        
        clean = torchaudio.functional.filtfilt(clean, self.a, self.b,)
        imu = torchaudio.functional.filtfilt(imu, self.a, self.b,)

        target_dB_FS = np.random.randint(-35, -15)
        noisy, clean, noise, scaler = snr_mix(noise, clean, snr, target_dB_FS, rir, eps=1e-6)
        imu = tailor_dB_FS(imu, target_dB_FS)[0]

        dvector = self.vectors[file]
        return {'imu': imu, 'clean': clean, 'vad': vad_annotation(clean), 'noisy': noisy, 'file': file, 'noise': noise, 'dvector': dvector}  
class EMSBDataset():
    def __init__(self, emsb, noise=None, mono=False, ratio=1, snr=(-5, 15), rir=None, length=5):
        self.dataset = []
        self.mono = mono
        self.ratio = ratio
        self.snr_list = np.arange(snr[0], snr[1], 1)
        sr = 16000
        with open(emsb, 'r') as f:
            data = json.load(f)
            left = data['left']; right = data['right']
            if ratio > 0:
                left = left[:int(len(data['left']) * self.ratio)]
                right = right[:int(len(data['right']) * self.ratio)]
            else:
                left = left[-int(len(data['left']) * self.ratio):]
                right = right[-int(len(data['right']) * self.ratio):]
            self.left_dataset = BaseDataset(left, sample_rate=sr, length=length)
            # self.right_dataset = BaseDataset(right, sample_rate=sr, length=length)
        self.noise_dataset = noise
        if self.noise_dataset is not None:
            self.noise_dataset = BaseDataset(noise, sample_rate=sr)
            self.noise_length = len(self.noise_dataset)
        self.rir = rir
        if self.rir is not None:
            with open(rir, 'r') as f:
                data = json.load(f)
            self.rir = data
            self.rir_length = len(self.rir)
        self.b, self.a = signal.butter(4, 100, 'highpass', fs=16000)
    def __len__(self):
        return len(self.left_dataset)
    def __getitem__(self, index):
        left, file = self.left_dataset[index]
        clean = left[:1, :]
        imu = left[1:, :]
        noise, _ = self.noise_dataset.__getitem__(np.random.randint(0, self.noise_length))

        if noise.shape[-1] <= clean.shape[-1]:
            noise = torch.nn.functional.pad(noise, (0, clean.shape[-1] - noise.shape[-1], 0, 0), 'constant')
        else:
            offset = np.random.randint(0, noise.shape[-1] - clean.shape[-1])
            noise = noise[:, offset:offset+clean.shape[-1]]

        use_reverb = False if self.rir is None else bool(np.random.random(1) < 0.75)
        rir = torchaudio.load(self.rir[np.random.randint(0, self.rir_length)][0])[0] if use_reverb else None
        snr = np.random.choice(self.snr_list)
        
        clean = torchaudio.functional.filtfilt(clean, self.a, self.b,)
        imu = torchaudio.functional.filtfilt(imu, self.a, self.b,)

        target_dB_FS = np.random.randint(-35, -15)
        noisy, clean, noise, scaler = snr_mix(noise, clean, snr, target_dB_FS, rir, eps=1e-6)
        imu = tailor_dB_FS(imu, target_dB_FS)[0]

        mixture = torch.cat([clean, noise], dim=0)
        return {'imu': imu, 'clean': clean, 'vad': vad_annotation(clean), 'noisy': noisy, 'file': file, 'noise': noise, 'mixture': mixture}  

class V2SDataset():
    def __init__(self, data, noise=None, snr=(-5, 15), rir=None, length=None):
        sr = 16000
        with open(data, 'r') as f:
            data = json.load(f)
            data_list = []
            for speaker in data.keys():
                for date in data[speaker]:
                    data_list += data[speaker][date]
        self.left_dataset = BaseDataset(data_list, sample_rate=sr, length=length)

        self.snr_list = np.arange(snr[0], snr[1], 1)
        self.rir = rir
        self.noise = noise
        sr = 16000
        self.b, self.a = signal.butter(4, 100, 'highpass', fs=16000)
        self.b = torch.from_numpy(self.b, ).to(dtype=torch.float)
        self.a = torch.from_numpy(self.a, ).to(dtype=torch.float)
    def __len__(self):
        return len(self.left_dataset)
    def __getitem__(self, index):
        left, file = self.left_dataset[index]
        noisy = left[:1, :]
        imu = left[1:, :]
        
        noisy = torchaudio.functional.filtfilt(noisy, self.a, self.b,)
        imu = torchaudio.functional.filtfilt(imu, self.a, self.b,)

        noisy *= 4; imu *= 4 # This is a magic number to move the dBFS from 38 to 26
        return {'imu': imu, 'clean': noisy, 'vad': vad_annotation(imu), 'noisy': noisy, 'file': file}