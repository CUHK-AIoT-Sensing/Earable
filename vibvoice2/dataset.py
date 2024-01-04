import json
import math
import numpy as np
import scipy.signal as signal
import torchaudio
import torch
from model.compress import flac_codec, mp3_codec

def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = (y ** 2).mean() ** 0.5
    scaler = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scaler
    return y, rms, scaler
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
def snr_mix(noise_y, clean_y, snr, target_dB_FS, rir=None, eps=1e-6):
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
            clean_y = torchaudio.functional.fftconvolve(clean_y, rir)[:len(clean_y)]
        clean_rms = (clean_y ** 2).mean() ** 0.5
        noise_rms = (noise_y ** 2).mean() ** 0.5
        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y
        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = target_dB_FS
        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar
        noise_y *= noisy_scalar
        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if (noise_y.max() > 0.999).any():
            noisy_y_scalar = (noisy_y).abs().max() / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar
            noise_y = noise_y / noisy_y_scalar
        return noisy_y, clean_y, noise_y, noisy_scalar

class BaseDataset:
    def __init__(self, files=None, pad=True, sample_rate=16000, length=5, stride=3, codec=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.sample_rate = sample_rate
        self.length = length
        self.stride = stride
        self.pad = True
        self.codec = codec
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
            if self.codec is not None:
                file = file.replace('Audio', self.codec).replace('wav', self.codec.split('_')[0])
            if self.length is None:
                data, _ = torchaudio.load(file)
                return data, file
            else:
                data, _ = torchaudio.load(file, frame_offset=self.stride * index * self.sample_rate, num_frames=self.length * self.sample_rate)
                if data.shape[-1] < (self.sample_rate * self.length):
                    pad_before = 0
                    pad_after = (self.sample_rate *self.length) - data.shape[-1]
                    # pad_before = np.random.randint((self.sample_rate *self.length) - data.shape[-1])
                    # pad_after = (self.sample_rate *self.length) - data.shape[-1] - pad_before
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
    def __init__(self, data, noise=None, snr=(-5, 15), rir=None, length=5, codec = 'mp3_16k'):
        self.snr_list = np.arange(snr[0], snr[1], 1)
        sr = 16000
        with open(data, 'r') as f:
            data = json.load(f)
            left = []
            for speaker in data.keys():
                left += data[speaker]
            self.codec_dataset = BaseDataset(left, sample_rate=sr, length=length, codec=codec)
            self.dataset = BaseDataset(left, sample_rate=sr, length=length)
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
        return len(self.dataset)
    def __getitem__(self, index):
        raw, raw_file = self.dataset[index]
        raw = raw[:1, :]
        left, file = self.codec_dataset[index]
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
        if rir is not None:
            raw = torchaudio.functional.fftconvolve(raw, rir)[:len(raw)]
        raw *= scaler
        return {'imu': imu, 'clean': clean, 'noisy': noisy, 'file': file, 'noise': noise, 'raw': raw}  
  
class EMSBDataset():
    def __init__(self, emsb, noise=None, mono=False, ratio=1, snr=(-5, 15), rir=None):
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
            self.left_dataset = BaseDataset(left, sample_rate=sr)
            self.right_dataset = BaseDataset(right, sample_rate=sr)
        self.noise_dataset = noise
        if self.noise_dataset is not None:
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
        noisy = left[1:, :]
        imu = left[:1, :]
        
        noisy = torchaudio.functional.filtfilt(noisy, self.a, self.b,)
        imu = torchaudio.functional.filtfilt(imu, self.a, self.b,)

        imu *= 4 # magic number to compensate two modality, based on sensor property
        noisy *= 4; imu *= 4 # This is a magic number to move the dBFS from 38 to 26
        return {'imu': imu, 'clean': noisy, 'vad': vad_annotation(imu), 'noisy': noisy, 'file': file}
    
if __name__ == "__main__":
    import scipy.io.wavfile as wavfile
    import os
    from tqdm import tqdm
    # directly run the script to save the dataset (noisy (2-channel: audio+imu), clean)
    rir = 'json/rir.json'
    dataset = 'ABCS'
    BATCH_SIZE = 1
    noises = [
              'json/ASR_aishell-dev.json',
              'json/other_DEMAND.json',
              ]
    noise_file = []
    for noise in noises:
        noise_file += json.load(open(noise, 'r'))

    datasets = [ABCSDataset('json/ABCS_train.json', noise=noise_file), 
                   ABCSDataset('json/ABCS_dev.json', noise=noise_file),
                   ABCSDataset('json/ABCS_test.json', noise=noise_file)]

    ori_folder = '../ABCS/Audio'
    Noisy_folder = '../ABCS_tmp/Noisy'
    Clean_folder = '../ABCS_tmp/Clean'
    for dataset in datasets:
        loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False, pin_memory=True)
        for sample in tqdm(loader):
            acc = sample['imu'][0].numpy(); noisy = sample['noisy'][0].numpy(); clean = sample['clean'][0].numpy(); file = sample['file'][0]
            noisy = np.concatenate([noisy, acc], axis=0)
            noisy_file = file.replace(ori_folder, Noisy_folder)
            clean_file = file.replace(ori_folder, Clean_folder)

            os.makedirs(os.path.dirname(noisy_file), exist_ok=True)
            os.makedirs(os.path.dirname(clean_file), exist_ok=True)
            wavfile.write(noisy_file, 16000, noisy.T)
            wavfile.write(clean_file, 16000, clean.T)
