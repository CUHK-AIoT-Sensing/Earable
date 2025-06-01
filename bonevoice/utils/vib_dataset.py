import json
import numpy as np
import torch
import os
import librosa
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def _EMSB_dataset(directory):
    g = os.walk(directory)
    count = 0; timer = 0
    data = {'left': [], 'right': []}
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
            data['left'].append(left)
            data['right'].append(right)
            timer += min(left_duration, right_duration)
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
}
V2S_config = {
    'vibration_channel': 1,
    'audio_channel': 0,
    'sample_rate': 16000,
    'gain': 4,
}
ABCS_config = {
    'vibration_channel': 1,
    'audio_channel': 0,
    'sample_rate': 16000
}

class VibDataset():
    def __init__(self, json_file, config, length=5):
        self.length = length
        self.config = config
        assert json_file.endswith('.json'), "Dataset should be a json file"
        with open(json_file, 'r') as f:
            self.json_file = json.load(f)
        # json: {split: [{'file': file_path, 'duration': duration}, ...]}
        self.dataset = []
        for split, data in self.json_file.items():
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
        if 'gain' in self.config:
            audio *= self.config['gain']; vibration *= self.config['gain']
        return {
            'audio': audio,
            'vibration': vibration,
        }
    
def EMSB_dataset(json_file=EMSB_json, config=EMSB_config, length=5):
    """
    Load the EMSB dataset from a json file.
    """
    return VibDataset(json_file, config, length)
def V2S_dataset(json_file=V2S_json, config=V2S_config, length=5):
    """
    Load the V2S dataset from a json file.
    """
    return VibDataset(json_file, config, length)
def ABCS_dataset(json_file=ABCS_json, config=ABCS_config, length=5):
    """
    Load the ABCS dataset from a json file.
    """
    return VibDataset(json_file, config, length)

if __name__ == "__main__":
    _EMSB_dataset('../dataset/Vibration/EMSB')
    _ABCS_dataset('../dataset/Vibration/ABCS')
    _V2S_dataset('../dataset/Vibration/V2S')
