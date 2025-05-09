'''
Dynamic dataset
1) don't have manual label, but motion capture label (if any)
2) don't have controlled speaker, overlap exist
3) have imu data, either by real-time recording or synthetic from motion capture
'''
from torch.utils.data import Dataset
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
import datetime




class Conversation_dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        subjects = os.listdir(root_dir) # ['01', '02']
        subjects.remove('meta.txt')
        self.subjects = subjects
        self.files = {}
        for subject in subjects:
            files = os.listdir(os.path.join(root_dir, subject))
            wav_files = [f for f in files if f.endswith('.wav')]; wav_files.sort()
            txt_files = [f for f in files if f.endswith('.txt')]; txt_files.sort()
            json_files = [f for f in files if f.endswith('.json')]; json_files.sort()
            npy_files = [f for f in files if f.endswith('.npy')]; npy_files.sort()
            self.files[subject] = {'wav': wav_files, 'txt': txt_files, 'json': json_files, 'npy': npy_files}


    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        data = {}
        data_folder = os.path.join(self.root_dir, self.data[idx])

        for subject in self.subjects:
            audio_file = os.path.join(data_folder, subject + '.wav')
            audio = librosa.load(audio_file, sr=None, mono=False)[0]
            data[subject] = audio
        return data
    
if __name__ == '__main__':
    dataset_folder = 'dataset/cuhk/2025-03-14'
    # preprocess_dataset(dataset_folder)
    dataset = Conversation_dataset(dataset_folder)