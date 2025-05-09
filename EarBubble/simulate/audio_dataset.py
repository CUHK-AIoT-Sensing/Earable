from torch.utils.data import Dataset, ConcatDataset, random_split
import os
import numpy as np
import librosa
import pandas as pd

import sys
sys.path.append('..')
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

speech_vad_model = load_silero_vad()


class FSD50K_dataset_wrapper(Dataset):
    def __init__(self, root='../dataset/', split='train', sr=16000):
        if split == 'train':
            self.dataset = FSD50K_dataset(root, split='dev')
        else:
            self.dataset = FSD50K_dataset(root, split='eval')
        self.dataset, _ = Singleclass_dataset(self.dataset)
        self.sr = sr
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        output_dict = self.dataset[idx]
        audio = output_dict['audio']
        label = output_dict['cls_label']; label = np.argmax(label)
        active_frame = np.ones(int(len(audio) / self.sr / 0.1))
        return audio, label, active_frame

class AudioSet_dataset_wrapper(Dataset):
    def __init__(self, root='../dataset/audioset', split='train', sr=16000):
        self.dataset = AudioSet_dataset(root, split=split, modality='audio', label_level='frame')
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        output_dict = self.dataset.__getitem__(idx)
        audio = output_dict['audio']
        label = output_dict['cls_label']
        active_frame = np.sum(label, axis=1) >= 1

        return audio, 0, active_frame

class NIGENS_dataset(Dataset):
    def __init__(self, root='NIGENS', split='train', sr=16000, duration=10):
        self.root_dir = root
        self.data = []
        self.sr = sr
        self.duration = duration
        self.class_names = ['alarm', 'baby', 'crash', 'dog', 'engine', 'femaleScream', 'femaleSpeech', 'fire', 'footsteps',
                            'knock', 'maleScream', 'maleSpeech', 'phone', 'piano',]
        for sound_class in (self.class_names):
            folder = os.path.join(self.root_dir, sound_class)
            files = os.listdir(folder)
            audio_files = [os.path.join(folder, file) for file in files if file.endswith('.wav')]
            txt_files = [os.path.join(folder, file) for file in files if file.endswith('.txt')]
            audio_files = [audio for audio in audio_files if audio + '.txt' in txt_files]
            txt_files = [audio + '.txt' for audio in audio_files]
            assert len(audio_files) == len(txt_files)
            
            sound_class = [sound_class] * len(audio_files)
            data = [(audio, txt, class_name) for audio, txt, class_name in zip(audio_files, txt_files, sound_class)]   
            if split == 'train':
                self.data += data[:int(len(data) * 0.8)]
            else:
                self.data += data[int(len(data) * 0.8):]   
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        audio, txt, class_name = self.data[idx]
        audio, sr = librosa.load(audio, sr=self.sr)
        active_frame = np.zeros(int(len(audio) / self.sr / 0.1))
        with open(txt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            start, end = line.strip().split()
            start_frame = int(float(start) / 0.1)
            end_frame = int(float(end) / 0.1)
            active_frame[start_frame:end_frame] = 1
        class_idx = self.class_names.index(class_name)

        if len(audio) > self.sr * self.duration:
            random_offset = np.random.randint(0, len(audio) - self.sr * self.duration)
            audio = audio[random_offset:random_offset + self.sr * self.duration]
            active_frame = active_frame[random_offset // int(self.sr * 0.1): (random_offset + self.sr * self.duration) // int(self.sr * 0.1)]
        else:
            audio = np.pad(audio, (0, self.sr * self.duration - len(audio)))
            active_frame = np.pad(active_frame, (0, self.sr * self.duration - len(active_frame)))
            
        return audio, class_idx, active_frame
    
class TIMIT_dataset(Dataset):
    def __init__(self, root='TIMIT', split='TRAIN', sr=16000):
        self.root_dir = os.path.join(root, split)
        self.data = []
        self.sr = sr
        for DR in os.listdir(self.root_dir):
            for P in os.listdir(os.path.join(self.root_dir, DR)):
                folder = os.path.join(self.root_dir, DR, P)
                files = os.listdir(folder)
                files = [os.path.join(folder, file) for file in files if file.endswith('.WAV')]
                self.data += (files)
        self.class_name = ['speech']
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        audio = librosa.load(self.data[idx], sr=self.sr)[0]
        audio_8000 = librosa.resample(y=audio, orig_sr=self.sr, target_sr=8000)
        speech_timestamps = get_speech_timestamps(
            audio_8000, speech_vad_model, threshold=0.5, sampling_rate=8000,  # VAD model, threshold and sampling rate
            return_seconds=True,  # Return speech timestamps in seconds (default is samples)
            )
        active_frames = np.zeros(int(len(audio) / self.sr / 0.1))
        for timestamp in speech_timestamps:
            start_frame = int(timestamp['start'] // 0.1)
            end_frame = int(timestamp['end'] // 0.1)
            active_frames[start_frame:end_frame] = 1
        # active_frames = np.ones(int(len(audio) / self.sr / 0.1))

        return audio, 0, active_frames
    
class VCTK_dataset(Dataset):
    def __init__(self, root, split='train', sr=16000):
        self.root_dir = os.path.join(root, 'wav48_silence_trimmed')
        self.data = []
        self.sr = sr
        speakers = os.listdir(self.root_dir)
        speakers = [speaker for speaker in speakers if speaker.startswith('p')]
        if split == 'train':
            speakers = speakers[:int(len(speakers) * 0.8)]
        else:
            speakers = speakers[int(len(speakers) * 0.8):]
        for speaker in speakers:
            speaker_folder = os.path.join(self.root_dir, speaker)
            for audio in os.listdir(speaker_folder):
                if audio.endswith('.flac'):
                    self.data.append(os.path.join(speaker_folder, audio))
        self.class_name = ['speech']
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        audio = librosa.load(self.data[idx], sr=self.sr)[0]
        active_frames = np.ones(int(len(audio) / self.sr / 0.1))
        return audio, 0, active_frames

def dataset_parser(dataset, relative_path, sr=16000):
    if dataset == 'TIMIT':
        root = os.path.join(relative_path, 'TIMIT')
        train_dataset = TIMIT_dataset(root=root, split='TRAIN', sr=sr)
        test_dataset = TIMIT_dataset(root=root, split='TEST', sr=sr)
    elif dataset == 'VCTK':
        root = os.path.join(relative_path, 'VCTK')
        train_dataset = VCTK_dataset(root=root, split='train', sr=sr)
        test_dataset = VCTK_dataset(root=root, split='test', sr=sr)
    elif dataset == 'NIGENS':
        root = os.path.join(relative_path, 'NIGENS')
        train_dataset = NIGENS_dataset(root=root, split='train', sr=sr)
        test_dataset = NIGENS_dataset(root=root, split='test', sr=sr)
    elif dataset == 'AudioSet':
        root = os.path.join(relative_path, 'audioset')
        train_dataset = AudioSet_dataset_wrapper(root=root, split='train', sr=sr)
        test_dataset = AudioSet_dataset_wrapper(root=root, split='eval', sr=sr)
    elif dataset == 'FSD50K':
        root = os.path.join(relative_path, 'FSD50K')
        train_dataset = FSD50K_dataset_wrapper(root=root, split='train', sr=sr)
        test_dataset = FSD50K_dataset_wrapper(root=root, split='test', sr=sr)
    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = dataset_parser('AudioSet', '.')

    # data = train_dataset[0]
    # dataset = NIGENS_dataset()
    # data = dataset[0]

