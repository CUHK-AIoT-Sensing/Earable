import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, ConcatDataset
from .fsd50k_dataset  import FSD50KDataset
import os
import scipy.signal as signal

class SNR_Controller():
    def __init__(self, snr, dBFS=(-35, -15)):
        self.snr = snr
        self.dBFS = dBFS

    def __call__(self, audio, noise, eps=1e-8):
        snr = np.random.uniform(self.snr[0], self.snr[1])
        # snr to float32
        snr = np.float32(snr)
        audio_rms = (audio ** 2).mean() ** 0.5
        noise_rms = (noise ** 2).mean() ** 0.5
        scale_factor = audio_rms / (noise_rms + eps)
        noise = noise * scale_factor / (10 ** (snr / 20))
        audio = audio + noise

        # Normalize the audio to the specified dBFS level
        # dBFS = np.random.uniform(self.dBFS[0], self.dBFS[1])
        # audio = audio / (np.max(np.abs(audio)) + eps) * (10 ** (dBFS / 20))
        # audio = np.clip(audio, -1.0, 1.0)
        return audio, noise, snr

def enroll_audio(folder):
    output_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac') or file.endswith('.WAV'):
                output_files.append(os.path.join(root, file))
    return output_files

class Conditional_SpeechEnhancementDataset(Dataset):
    def __init__(self, dataset, noise_folders, length=5, sample_rate=16000, snr_range=(-5, 15), rir=None):
        """
        A dataset for speech enhancement tasks, combining clean speech with noise.
        Args:
            (Dataset): A dataset containing clean speech samples.
                the dataset should return a dictionary with 'audio' key for audio data (other modalities are accepted).
            noises_folders (list): list of folders containing noise files.
        """
        self.clean_dataset = dataset

        self.length = length
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.snr_controller = SNR_Controller(snr_range, dBFS=(-35, -15))

        self.noise_dataset = FSD50KDataset(folder='../dataset/Audio/FSD50K', split='dev', length=length, sample_rate=sample_rate, feature=True)

        if rir is not None:
            self.rir_files = enroll_audio(rir)
        else:
            self.rir_files = None

    def __len__(self):
        return len(self.clean_dataset)
    def __getitem__(self, index):
        use_reverb = False if self.rir_files is None else bool(np.random.random(1) < 0.75)
        if use_reverb:
            # Randomly select a RIR file from the list
            rir_file = np.random.choice(self.rir_files)
            rir = librosa.load(rir_file, sr=self.sample_rate, mono=True)[0]
        else:
            rir = None

        data = self.clean_dataset[index] # {'vibration': vibration, 'audio': audio}

        random_index = np.random.randint(0, len(self.noise_dataset))
        noise_data = self.noise_dataset[random_index]  # Get noise data: {'filename': ..., 'label_names': ..., 'mids': ..., 'idx': ..., 'one_hot': ..., 'audio': ...}
        noise = noise_data['audio']; feature = noise_data['feature']
        data['feature'] = feature  # Add noise feature to the data

        if rir is not None:
            data['audio'] = signal.fftconvolve(data['audio'], rir)[:len(data['audio'])]  # Apply RIR to the clean audio

        noisy_data, noise, snr = self.snr_controller(data['audio'], noise)
        data['noisy_audio'] = noisy_data
        data['dualchannel'] = np.stack([noisy_data, data['vibration']], axis=0)  # Dual-channel audio
        data['snr'] = snr
        data['noise'] = noise
        return data
