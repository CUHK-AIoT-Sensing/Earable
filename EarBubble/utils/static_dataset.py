'''
Static dataset is formed by segments of audio recordings and corresponding log files (the location of the speakers).
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

def preprocess_dataset(dataset_folder):
    """
    Preprocess the dataset by segmenting audio files and saving them into a structured format.
    """
    channels = 2
    sr = 44100
    subjects = ['01', '02']
    audios = {}
    audios_start_time = {}
    for subject in subjects:
        files = os.listdir(os.path.join(dataset_folder, subject))
        files.sort()
        start_time = datetime.datetime.strptime(files[0], '%Y-%m-%d_%H-%M-%S-%f.wav')
        audios_start_time[subject] = start_time
        audios[subject] = []
        
        for file in tqdm(files):
            audio, _ = librosa.load(os.path.join(dataset_folder, subject, file), sr=None, mono=False)
            start_time_segment = datetime.datetime.strptime(file, '%Y-%m-%d_%H-%M-%S-%f.wav')
            gap = ((start_time_segment - start_time).total_seconds() * sr)
            if gap > 0:
                audios[subject].append(np.zeros((channels, int(gap))))
            audios[subject].append(audio)
            start_time = start_time_segment + datetime.timedelta(seconds=audio.shape[1] / sr)
        audios[subject] = np.concatenate(audios[subject], axis=1)
        print('Subject {} has {} seconds of audio'.format(subject, audios[subject].shape[1] / sr))

    log_files = os.listdir(os.path.join(dataset_folder, 'log'))
    for log_file in tqdm(log_files):
        log = pd.read_csv(os.path.join(dataset_folder, 'log', log_file), dtype=str)
        log_time = datetime.datetime.strptime(log_file, '%Y-%m-%d_%H-%M-%S-%f.csv')
        recording_time = int(log['time'].values[0])
        segment_folder = os.path.join(dataset_folder, 'segments', log_file.split('.')[0])
        os.makedirs(segment_folder, exist_ok=True)

        for subject in subjects:
            recording_time_subject = (log_time - audios_start_time[subject]).total_seconds()
            audio_segment = audios[subject][:, int(recording_time_subject * sr):int((recording_time_subject + recording_time) * sr)]
            audio_segment_file = os.path.join(segment_folder, subject + '.wav')
            sf.write(audio_segment_file, audio_segment.T, sr)
        
        log_file_segment = os.path.join(segment_folder, 'log.csv')
        log.to_csv(log_file_segment, index=False)

def construct_conversation(data_folder):
    '''
    inspect the log files, find the corresponding recordings with the same location and orientation
    '''
    log_files = os.listdir(os.path.join(data_folder, 'log'))
    locations = []
    for log_file in log_files:
        log = pd.read_csv(os.path.join(data_folder, 'log', log_file), dtype=str)
        location = log[['location', 'orientation']].values.tolist()
        locations.append((location, log_file))

    conversations = []
    for i in range(len(locations)):
        location_i, log_file_i = locations[i]
        for j in range(i + 1, len(locations)):
            location_j, log_file_j = locations[j]
            if location_i == location_j:
                conversations.append((log_file_i, log_file_j))
    # save in txt
    with open(os.path.join(data_folder, 'conversation.txt'), 'w') as f:
        for conversation in conversations:
            f.write(conversation[0] + ' ' + conversation[1] + '\n')
    
class Segment_dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.segments_dir = os.path.join(root_dir, 'segments')
        self.data = os.listdir(self.segments_dir)
        self.subjects = ['01', '02']
        conversation = os.path.join(root_dir, 'conversation.txt')
        self.conversations = []
        with open(conversation, 'r') as f:
            self.conversations = f.readlines()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = {}
        data_folder = os.path.join(self.segments_dir, self.data[idx])
        log = pd.read_csv(os.path.join(data_folder, 'log.csv'), dtype=str)

        for subject in self.subjects:
            audio_file = os.path.join(data_folder, subject + '.wav')
            audio = librosa.load(audio_file, sr=None, mono=False)[0]
            # log_subject = log[log['type'] == subject]
            data[subject] = audio
        return data, log

if __name__ == '__main__':
    preprocess_dataset('dataset/cuhk/2025-03-07')
    # construct_conversation('dataset/cuhk/2025-03-07')

    dataset = Segment_dataset('dataset/cuhk/2025-03-07')