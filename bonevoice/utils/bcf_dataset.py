from .bcf import Bone_Conduction_Function, estimate_frequency_response
from torch.utils.data import Dataset
import os
import librosa
import numpy as np


def enroll_audio(folder):
    output_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac') or file.endswith('.WAV'):
                output_files.append(os.path.join(root, file))
    return output_files

class aishellDataset():
    def __init__(self, folder, split='all', length=5, sample_rate=16000):
        self.folder = folder
        self.split = f'aishell-{split}'
        self.length = length
        self.sample_rate = sample_rate
        self.dataset = []
        self.split_folder = os.path.join(folder, self.split)
        self.audio_files = enroll_audio(self.split_folder)
    def __len__(self):
        return len(self.audio_files)
    def __getitem__(self, index):
        file = self.audio_files[index]
        audio = librosa.load(file, sr=self.sample_rate, mono=True)[0]  # Load audio file
        if len(audio) <= self.length * self.sample_rate:
            pad = (self.length * self.sample_rate) - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant')
        else:
            offset = np.random.randint(0, len(audio) - self.length * self.sample_rate)
            audio = audio[offset:offset + self.length * self.sample_rate]
        data = {
            'index': index,
            'audio': audio,
        }
        return data

class BCFAugmentationDataset():
    '''
    Dataset for augmenting audio data with bone conduction function.
    data['vibration'] will be augmented with bone conduction function + clean audio.
    '''
    def __init__(self, audio_dataset, bcf_dataset='ABCS'):
        self.audio_dataset = audio_dataset
        self.bcf = Bone_Conduction_Function(bcf_dataset)
    def __len__(self):
        return len(self.audio_dataset)
    def __getitem__(self, index):
        data = self.audio_dataset[index]
        bcf, augmented_vibration = self.bcf.predict(data['audio'])
        augmented_vibration = np.float32(augmented_vibration)
        data['vibration'] = augmented_vibration
        return data

class BCFNormVibrationDataset(Dataset):
    '''
    Dataset that convert vibration into low-frequency audio using bone conduction function as condition.
    for test purpose, use the real vib_dataset has 'vibration', 'audio'
    for train purpose, use the synthetic vib_dataset has only 'audio' - BCF is generated randomly, the 'vibration' is generated
    compared to BCFAugmentationDataset, this dataset has 'vibration', 'audio' and 'bcf' keys.
    '''
    def __init__(self, dataset, real_vib=True):
        self.dataset = dataset
        self.real_vib = real_vib
        if not real_vib:
            self.bcf = Bone_Conduction_Function('random')
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        data = self.dataset[index]
        if self.real_vib:
            freqs, bcf = estimate_frequency_response(data['vibration'])
            data['bcf'] = bcf
        else:
            bcf, augmented_vibration = self.bcf.predict(data['audio'])
            augmented_vibration = np.float32(augmented_vibration)
            data['vibration'] = augmented_vibration
            data['bcf'] = bcf
        return data