import json
import numpy as np
import torch
import os
import librosa
import warnings
import scipy.signal as signal
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

def _EMSB_dataset(directory):
    g = os.walk(directory)
    count = 0; timer = 0
    data = {}
    for path, dir_list, file_list in g:
        if len(file_list) == 2: # left + right
            left_file = os.path.join(path, file_list[0]); right_file = os.path.join(path, file_list[1])
            left_duration = librosa.get_duration(path=left_file); right_duration = librosa.get_duration(path=right_file)
            left = {
                'file': left_file,
                'duration': left_duration,
            }
            right = {
                'file': right_file,
                'duration': right_duration
            }
            data[path] = []
            data[path].append(left)
            data[path].append(right)
            timer += left_duration + right_duration
            count += 1
    print('EMSB dataset, we have recording:', count, 'whole duration (hour):', timer/3600)
    json.dump(data, open('data/EMSB.json', 'w'), indent=4)

def _ABCS_dataset(directory):
    splits = ['train', 'dev', 'test']
    data = {}
    count = 0; timer = 0
    for split in splits:
        split_path = directory + '/Audio/' + split
        for speaker in os.listdir(split_path):
            path = split_path + '/' + speaker
            data[speaker] = []
            for f in os.listdir(path):
                filename = os.path.join(path, f)
                duration = librosa.get_duration(path=filename)
                data[speaker].append({
                    'file': filename,
                    'duration': duration
                })
                timer += duration
                count += 1
    print('ABCS datast', 'we have recording:', count, 'whole duration (hour):', timer/3600)
    json.dump(data, open('data/ABCS.json', 'w'), indent=4)

def _V2S_dataset(directory):
    # ban warning: this function will take a long time to run
    data = {}
    speakers = os.listdir(directory)
    count = 0; timer = 0
    for speaker in speakers:
        path = directory + '/' + speaker
        for date in os.listdir(path):
            date_path = path + '/' + date
            files = os.listdir(date_path)
            data[date_path] = []
            for f in files:
                try:
                    filename = os.path.join(date_path, f)
                    duration = librosa.get_duration(path=filename)

                    data[date_path].append({
                        'file': filename,
                        'duration': duration
                    })
                    timer += duration
                    count += 1
                except:
                    pass
    print('V2S dataset we have recordings:', count, 'whole duration (hour):', timer/3600)
    json.dump(data, open('data/V2S.json', 'w'), indent=4)

EMSB_json = 'data/EMSB.json'
V2S_json = 'data/V2S.json'
ABCS_json = 'data/ABCS.json'
EMSB_config = {
    'vibration_channel': 1,
    'audio_channel': 0,
    'sample_rate': 16000,
    'highpass': 100,
    'norm_vibration': 2000,
}
V2S_config = {
    'vibration_channel': 1,
    'audio_channel': 0,
    'sample_rate': 16000,
    'gain': 4,
    'highpass': 100,
    'norm_vibration': 2000,
    'text': True,  # whether to load the text labels

}
ABCS_config = {
    'vibration_channel': 1,
    'audio_channel': 0,
    'sample_rate': 16000,
    'highpass': 100,
    'norm_vibration': 2000,
}

class VibDataset():
    def __init__(self, json_file, config, split='all', length=5):
        self.length = length
        self.config = config
        assert json_file.endswith('.json'), "Dataset should be a json file"
        with open(json_file, 'r') as f:
            self.json_file = json.load(f)
        # json: {split: [{'file': file_path, 'duration': duration}, ...]}
        
        if 'highpass' in self.config:
            self.b, self.a = signal.butter(4, self.config['highpass'], 'highpass', fs=16000)
        if 'norm_vibration' in self.config:
            self.b, self.a = signal.butter(4, self.config['norm_vibration'], 'lowpass', fs=16000)
        self.split_dataset(split)
        
    def split_dataset(self, split):
        """
        if split == None, return the whole dataset
        if split == str of list of strings, return the dataset with the specified splits
        """
        if split == 'all':
            split_selected = list(self.json_file.keys())
        elif isinstance(split, str):
            split_selected = [split]
        elif isinstance(split, list):
            split_selected = split
        else:
            raise ValueError("split should be None, str or list of str")
        
        self.dataset = []
        for split in split_selected:
            data = self.json_file[split]
            for item in data:
                file = item['file']; duration = item['duration']
                num_examples = int((duration+self.length-1) // self.length)
                for i in range(num_examples):
                    start = i * self.length
                    end = start + self.length
                    self.dataset.append({
                        'file': file,
                        'start': start,
                        'end': end,
                    })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        file = data['file']; start = data['start']; end = data['end']

        audio, sr = librosa.load(file, sr=self.config['sample_rate'], mono=False, offset=start, duration=self.length)
        if audio.shape[-1] < self.config['sample_rate'] * self.length:
            pad = (self.config['sample_rate'] * self.length) - audio.shape[-1]
            audio = np.pad(audio, ((0, 0), (0, pad)), mode='constant')

        vibration = audio[self.config['vibration_channel'], :]
        audio = audio[self.config['audio_channel'], :]
        if 'highpass' in self.config:
            vibration = signal.filtfilt(self.b, self.a, vibration).copy().astype(np.float32)

        if 'norm_vibration' in self.config:
            norm_vibration = signal.filtfilt(self.b, self.a, audio).copy().astype(np.float32)

        if 'gain' in self.config:
            audio *= self.config['gain']; vibration *= self.config['gain']

        if 'text' in self.config:
            file_dir_name, file_name = os.path.dirname(file), os.path.basename(file)
            text_file = os.path.join(file_dir_name, 'labels.txt')
            labels = open(text_file, 'r').readlines()
            text = ''
            for label in labels:
                filename = label.split()[0]
                if filename == file_name[:-4]:
                    text = ' '.join(label.split()[1:])
                    break
            
        return {
            'audio': audio,
            'vibration': vibration,
            'norm_vibration': norm_vibration,
            'index': index,
            'text': text if 'text' in self.config else '',
        }
    
def EMSB_dataset(json_file=EMSB_json, config=EMSB_config, split='all', length=5):
    """
    Load the EMSB dataset from a json file.
    """
    return VibDataset(json_file, config, split, length)
def V2S_dataset(json_file=V2S_json, config=V2S_config, split='all', length=5):
    """
    Load the V2S dataset from a json file.
    """
    return VibDataset(json_file, config, split, length)
def ABCS_dataset(json_file=ABCS_json, config=ABCS_config, split='all', length=5):
    """
    Load the ABCS dataset from a json file.
    """
    return VibDataset(json_file, config, split, length)

if __name__ == "__main__":
    _EMSB_dataset('../dataset/Vibration/EMSB')
    _ABCS_dataset('../dataset/Vibration/ABCS')
    _V2S_dataset('../dataset/Vibration/V2S')
