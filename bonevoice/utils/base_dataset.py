import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, ConcatDataset
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

class SpeechEnhancementDataset(Dataset):
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

        self.noise_files = {}
        for noise_folder in noise_folders:
            self.noise_files[noise_folder] = enroll_audio(noise_folder)
        # find all noise files in the specified folder
        length_noise = [len(self.noise_files[noise_folder]) for noise_folder in noise_folders]
        print(f"Number of noise files in each folder: {length_noise}")

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
        noise_source = np.random.choice(list(self.noise_files.keys())) # Randomly select a noise source from the available noise folders
        if noise_source == 'self': 
            noise = self.clean_dataset[np.random.randint(0, len(self.clean_dataset) - 1)]['audio']  # Randomly select a noise sample from the same dataset
        else:
            noise_files = self.noise_files[noise_source]
            noise_path = np.random.choice(noise_files)

            noise = librosa.load(noise_path, sr=self.sample_rate, mono=True)[0]
            if len(noise) <= self.sample_rate * self.length:
                noise = np.pad(noise, (0, self.sample_rate * self.length - len(noise)), mode='constant')
            else:
                noise = noise[:self.sample_rate * self.length]

            if rir is not None:
                data['audio'] = signal.fftconvolve(data['audio'], rir)[:len(data['audio'])]  # Apply RIR to the clean audio

        noisy_data, noise, snr = self.snr_controller(data['audio'], noise)
        data['noisy_audio'] = noisy_data
        data['snr'] = snr
        data['noise'] = noise
        return data
