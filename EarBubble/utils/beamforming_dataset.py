from torch.utils.data import Dataset
import os
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from random import sample

from utils.simulate.parameter import SPEED_OF_SOUND, SMARTGLASS

def shift_mixture(input_data, target_position, sr, inverse=False):
    """
    Shifts the input according to the voice position. This
    lines up the voice samples in the time domain coming from a target_angle
    Args:
        input_data - M x T numpy array or torch tensor
        target_position - The location where the data should be aligned
        mic_radius - In meters. The number of mics is inferred from
            the input_Data
        sr - Sample Rate in samples/sec
        inverse - Whether to align or undo a previous alignment

    Returns: shifted data and a list of the shifts
    """
    # elevation_angle = 0.0 * np.pi / 180
    # target_height = 3.0 * np.tan(elevation_angle)
    # target_position = np.append(target_position, target_height)

    num_channels = input_data.shape[0]

    # Must match exactly the generated or captured data
    # mic_array = [[
    #     mic_radius * np.cos(2 * np.pi / num_channels * i),
    #     mic_radius * np.sin(2 * np.pi / num_channels * i),
    # ] for i in range(num_channels)]
    mic_array = SMARTGLASS.T[:, :2]
    # Mic 0 is the canonical position
    distance_mic0 = np.linalg.norm(mic_array[0] - target_position)
    shifts = [0]

    # Check if numpy or torch
    if isinstance(input_data, np.ndarray):
        shift_fn = np.roll
    elif isinstance(input_data, torch.Tensor):
        shift_fn = torch.roll
    else:
        raise TypeError("Unknown input data type: {}".format(type(input_data)))

    # Shift each channel of the mixture to align with mic0
    for channel_idx in range(1, num_channels):
        distance = np.linalg.norm(mic_array[channel_idx] - target_position)
        distance_diff = distance - distance_mic0
        shift_time = distance_diff / SPEED_OF_SOUND
        shift_samples = int(round(sr * shift_time))
        if inverse:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               shift_samples)
        else:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               -shift_samples)
        shifts.append(shift_samples)

    return input_data, shifts

def pick_source(label, source_audio):
    source_active_azimuth = np.zeros((len(source_audio), 360), dtype=np.int32)
    for l in label:
        azimuth = l[3]; source = l[2]
        azimuth_min = azimuth - 5; azimuth_max = azimuth + 5
        if azimuth_min < 0:
            source_active_azimuth[source, azimuth_min + 360:360] = 1
            source_active_azimuth[source, 0:azimuth_max] = 1
        elif azimuth_max >= 360:
            source_active_azimuth[source, azimuth_min:360] = 1
            source_active_azimuth[source, 0:azimuth_max - 360] = 1
        else:
            source_active_azimuth[source, azimuth_min:azimuth_max] = 1
    if np.random.uniform() < 0 or len(source_audio) < 1 or len(label) < 1:
        # return index where the source is not active
        postive_azimuth = np.max(source_active_azimuth, axis=0)
        negative_azimuth = np.where(postive_azimuth == 0)[0]
        azimuth = np.random.choice(negative_azimuth)
        source_idx = -1
    else:
        postive_azimuth = np.max(source_active_azimuth, axis=0)
        postive_azimuth = np.where(postive_azimuth)[0]
        azimuth = np.random.choice(postive_azimuth)
        # source_idx = source_active_azimuth[:, azimuth] == 1
        source_idx = np.argmax(source_active_azimuth[:, azimuth])
    return source_idx, azimuth

def get_dvector(audio_name, max_sources):
    # given direction (by default azimuth 90), extract the dvector (by default 256 dim)
    source_files = os.listdir(audio_name[:-4]); source_files = [f for f in source_files if f.endswith('.npy')]
    dvectors = []
    for i in range(max_sources):
        if i < len(source_files):
            dvector = np.load(os.path.join(audio_name[:-4], source_files[i]),).astype(np.float32)
        else:
            dvector = np.zeros(256, dtype=np.float32)
        dvectors.append(dvector)
    dvectors = np.array(dvectors)
    return dvectors

def random_dvector(mixture_audio, source_audio, source_vector, label, config):
    sr = config['sample_rate']
    source_idx, azimuth = pick_source(label, source_vector)
    if source_idx == None:
        source_dvector = np.zeros(256, dtype=np.float32)
        source_audio = np.zeros_like(mixture_audio)
    else:
        source_dvector = source_vector[source_idx]  
        source_audio = source_audio[source_idx]  
    target_pos = np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))])
    mixture_audio, shifts = shift_mixture(mixture_audio, target_pos, sr)
    return mixture_audio, source_audio, source_dvector 

def beamforming(mixture_audio, source_audio, label, config):
    sr = config['sample_rate']
    source_idx, azimuth = pick_source(label, source_audio)

    if source_idx == None:
        source_audio = np.zeros_like(mixture_audio)
    else:
        # mask the source_audio by source_idx
        source_audio = source_audio[source_idx]
        source_audio = np.sum(source_audio, axis=0, keepdims=True).astype(np.float32)

    target_pos = np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))])
    mixture_audio, shifts = shift_mixture(mixture_audio, target_pos, sr)
    return mixture_audio, source_audio

def region_beamforming(mixture_audio, source_audio, label, config):
    number_of_regions = config['num_region'] # note that it is the number of regions, speakers < regions
    regions = np.linspace(0, 360, number_of_regions + 1)
            
    region_audio = np.zeros((number_of_regions, config['sample_rate'] * config['duration']), dtype=np.float32)
    region_active = np.zeros(number_of_regions, dtype=np.int32)
    # label: [frame, class, source/instance, azimuth, elevation, distance]
    for i, source in enumerate(source_audio):
        if len(label) == 0: # no label
            continue
        # pick the source_i from label, get the average location, for fixed object, no need to average
        source_i_loc = label[label[:, 2] == i, 3:6]
        if len(source_i_loc) == 0:
            continue
        else:
            source_i_loc = np.mean(source_i_loc, axis=0)
            azimuth, elevation, distance = source_i_loc
            region_idx = np.digitize(azimuth, regions) - 1
            region_active[region_idx] = 1
            region_audio[region_idx] += source[0] # left channel
    return mixture_audio, region_audio

class Beamforming_dataset(Dataset):
    def __init__(self, root_dir, config=None):
        self.config = config
        self.root_dir = root_dir
        self.data_folder = os.path.join(self.root_dir, 'audio')
        self.label_folder = os.path.join(self.root_dir, 'meta')
        self.labels = os.listdir(self.label_folder)
        self.sr = self.config['sample_rate']
        self.duration = self.config['duration']
        self.output_format = self.config['output_format']
        self.max_sources = self.config['max_sources']
        self.framewise_meta() 
    def framewise_meta(self):
        '''
        split the full audio into small chunks, defined by the duration
        '''
        self.crop_labels = []
        for i, label_name in enumerate(tqdm(self.labels)):
            label = np.loadtxt(os.path.join(self.label_folder, label_name), delimiter=',', dtype=np.int32, skiprows=1)
            if len(label.shape) == 1:
                continue
            if label.shape[0] == 1:
                label = label[np.newaxis, :]
            audio_file = os.path.join(self.data_folder, label_name[:-4] + '/0.wav')
            max_frame = int(librosa.get_duration(path=audio_file) * 10)
            frame_duration = int(self.duration * 10)
            for start_frame in range(0, max_frame, frame_duration):
                end_frame = min(start_frame + frame_duration, max_frame)
                mini_chunk = []
                for l in label:
                    if l[0] >= start_frame and l[0] < end_frame:
                        mini_chunk.append(l)
                mini_chunk = np.array(mini_chunk)
                if len(mini_chunk) > 0: 
                    self.crop_labels.append((label_name, start_frame, end_frame, mini_chunk))
        print('Total crop labels:', len(self.crop_labels))
    def __len__(self):
        return len(self.crop_labels)
    def __getitem__(self, index): 

        label_name, start_frame, end_frame, label = self.crop_labels[index]
        start_frame_audio = start_frame / 10
        audio_name = os.path.join(self.data_folder, label_name)
        source_audio = []
        source_files = os.listdir(audio_name[:-4]); source_files = [f for f in source_files if f.endswith('.wav')]; source_files = sorted(source_files)
        for i in range(self.max_sources):
            if i < len(source_files):
                source_file = source_files[i]
                source, sr = librosa.load(os.path.join(audio_name[:-4], source_file), sr=self.sr, mono=False, 
                                        offset=start_frame_audio, duration=self.duration)
                if source.shape[-1] < self.duration * self.sr:
                    source = np.pad(source, ((0, 0), (0, self.duration * self.sr - source.shape[-1])))
            else:
                n_channel = source.shape[0]
                source = np.zeros((n_channel, self.duration * self.sr))
            source_audio.append(source)
        source_audio = np.array(source_audio)

        mixture_audio = np.sum(source_audio, axis=0).astype(np.float32)
        if self.output_format == 'raw':
            # keep the original format
            pass  
        elif self.output_format == 'separation':
            # fix to the first channel
            mixture_audio = mixture_audio[0] # [left, T]
            source_audio = source_audio[:, 0] # [source, left, T]            
        elif self.output_format == 'beamforming':
            mixture_audio, source_audio = beamforming(mixture_audio, source_audio, label, self.config)
        elif self.output_format == 'region':
            mixture_audio, source_audio = region_beamforming(mixture_audio, source_audio, label, self.config)
        elif self.output_format == 'dvector_distill':
            dvectors = get_dvector(audio_name, self.max_sources)
            mixture_audio, source_audio, source_dvector = random_dvector(mixture_audio, source_audio, dvectors, label, self.config)
            source_audio = source_dvector
        elif self.output_format == 'dvector_filter':
            dvectors = get_dvector(audio_name, self.max_sources)
            mixture_audio, source_audio, source_dvector = random_dvector(mixture_audio, source_audio, dvectors, label, self.config)
            mixture_audio = (mixture_audio[:1, :], source_dvector)
            source_audio = source_audio[:1, :]
        return mixture_audio, source_audio
