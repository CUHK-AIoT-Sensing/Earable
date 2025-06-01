import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, ConcatDataset
import os
from .vib_dataset import ABCS_dataset, EMSB_dataset, V2S_dataset

class SNR_Controller():
    def __init__(self, snr, dBFS):
        self.snr = snr
        self.dBFS = dBFS

    def __call__(self, audio, noise):
        snr = np.random.uniform(self.snr[0], self.snr[1])
        noise = noise / np.linalg.norm(noise) * np.linalg.norm(audio) / (10 ** (snr / 20))
        noise = noise[:audio.shape[0]]
        audio = audio + noise

        # Normalize the audio to the specified dBFS level
        dBFS = np.random.uniform(self.dBFS[0], self.dBFS[1])
        audio = audio / np.max(np.abs(audio)) * (10 ** (dBFS / 20))
        audio = np.clip(audio, -1.0, 1.0)
        return audio

def dataset_parser(dataset_names):
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'ABCS':
            datasets.append(ABCS_dataset())
        elif dataset_name == 'EMSB':
            datasets.append(EMSB_dataset())
        elif dataset_name == 'V2S':
            datasets.append(V2S_dataset())
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
    return ConcatDataset(datasets)

class SpeechEnhancementDataset(Dataset):
    def __init__(self, dataset_names, noises_folders, length=5, sample_rate=16000, snr_range=(-5, 15)):
        """
        A dataset for speech enhancement tasks, combining clean speech with noise.
        Args:
            dataset_names (list): List of dataset names to be combined, e.g., ['ABCS', 'EMSB', 'V2S'].
                (Dataset): A dataset containing clean speech samples.
                    the dataset should return a dictionary with 'audio' key for audio data (other modalities are accepted).
            noises_folders (list): list of folders containing noise files.
        """
        self.dataset_names = dataset_names
        self.clean_dataset = dataset_parser(dataset_names)

        self.length = length
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.snr_controller = SNR_Controller(snr_range, dBFS=(-35, -15))
        self.noise_files = []
        # find all noise files in the specified folder
        for noises_folder in noises_folders:
            for root, dirs, files in os.walk(noises_folder):
                for file in files:
                    if file.endswith('.wav') or file.endswith('.flac'):
                        self.noise_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.clean_dataset)
    def __getitem__(self, index):
        data = self.clean_dataset[index] # {'vibration': vibration, 'audio': audio}

        noise_path = np.random.choice(self.noise_files)
        noise = librosa.load(noise_path, sr=self.sample_rate, mono=True)[0]
        if len(noise) <= self.sample_rate * self.length:
            noise = np.pad(noise, (0, self.sample_rate * self.length - len(noise)), mode='constant')
        else:
            random_start = np.random.randint(0, len(noise) - self.sample_rate * self.length)
            noise = noise[random_start: random_start + self.sample_rate * self.length]
        noisy_data = self.snr_controller(data['audio'], noise)
        data['noisy_audio'] = noisy_data
        return data
