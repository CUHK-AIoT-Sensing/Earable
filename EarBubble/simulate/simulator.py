import matplotlib.pyplot as plt
import numpy as np
import sofa
import os
from random import uniform, sample 
import scipy.signal as signal
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import pyroomacoustics as pra
from hrtf_utils import HRTF_simulator
import json
from parameter import EARPHONE, SMARTGLASS

def random_room(num_room=200):
    rooms = []
    for _ in range(num_room):
        width = uniform(5, 15)
        length = uniform(5, 20)
        height = uniform(2, 3)
        room_dim = [width, length, height]
        rt60 = uniform(0.25, 0.75)
        # We invert Sabine's formula to obtain the parameters for the ISM simulator
        absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        rooms.append([room_dim, absorption, max_order])
    return rooms
def azimuth_to_rotation_matrix(radians):
    # Convert azimuth from degrees to radians

    # Create the rotation matrix for rotation around the Z-axis
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians), np.cos(radians), 0],
        [0, 0, 1]
    ])
    return rotation_matrix

def random_source(room_dim, mic_center, max_source, min_diff=45):
    positions = np.array([])
    azimuths, elevations, ranges = np.array([]), np.array([]), np.array([])
    while 1:
        position = np.random.rand(3) * np.array(room_dim)
        # Calculate the azimuth and elevation of the source
        azimuth = np.arctan2(position[1] - mic_center[1], position[0] - mic_center[0])
        elevation = np.arctan2(position[2] - mic_center[2], np.linalg.norm(position[:2] - mic_center[:2]))
        # keep the azimuth difference at least min_diff
        if len(azimuths) > 0:
            if np.min(np.abs(azimuth - azimuths)) < np.radians(min_diff):
                continue
        positions = np.append(positions, position); azimuths = np.append(azimuths, azimuth)
        elevations = np.append(elevations, elevation); ranges = np.append(ranges, np.linalg.norm(position - mic_center))        

        if len(azimuths) == max_source:
            break
    doas = np.array([azimuths, elevations]).T
    # doas to degrees
    doas = np.degrees(doas)
    positions = positions.reshape(-1, 3)
    return positions, doas, ranges

class simulator():
    def __init__(self, device, sr) -> None:
        if device == 'smartglass':
            self.mic_array = SMARTGLASS
            self.num_channel = np.shape(self.mic_array)[1]
            self.HRTF = False
            self.Reverb = True
        elif device == 'earphone':
            self.mic_array = EARPHONE
            self.HRTF = True
            self.Reverb = True
            self.num_channel = 2
            self.hrtf_simulator = HRTF_simulator()
        
        self.sr = sr 
        self.snr_lb, self.snr_ub = 20, 30
        self.offset = 0.5
        self.min_diff = 45
        self.room_dims = random_room()
    
    def _simulate(self, signals, room, source_locs, directivity=None):
        # TODO: how to set the accurate directivity?
        if directivity is None:
            dir_obj = None
        else:
            dir_obj = pra.directivities.CardioidFamily(
            orientation=pra.directivities.DirectionVector(azimuth=90, colatitude=15, degrees=True), p=0.5)
        
        if self.Reverb:
            for (source_loc, s) in zip(source_locs, signals):
                room.add_source(source_loc, signal=s, directivity=dir_obj)
            signals = room.simulate(return_premix=True)
        else: # reverb off, skip the room geometry
            signals = np.array(signals)
            signals = np.repeat(signals[:, np.newaxis, :], self.num_channel, axis=1)
        return signals

    def sound_event(self, save_folder, dataset, num_data, config):
        max_source = config['max_source']
        audio_folder = save_folder + '/audio'; meta_folder = save_folder + '/meta'
        os.makedirs(audio_folder, exist_ok=True); os.makedirs(meta_folder, exist_ok=True)

        if num_data is None:
            num_data = len(dataset)
        else: 
            num_data = num_data
        for i in tqdm(range(num_data)):
            room_dim, absorption, max_order = sample(self.room_dims, 1)[0]
            room = pra.ShoeBox(room_dim, fs=self.sr, max_order=max_order, absorption=absorption)

            mic_center = np.array([uniform(0 + self.offset, room_dim[0] - self.offset), 
                                   uniform(0 + self.offset, room_dim[1] - self.offset), uniform(1.5, 1.8)])
            room.add_microphone_array(mic_center[:, np.newaxis] + self.mic_array)

            source_locs, doas, ranges = random_source(room_dim, mic_center, max_source, self.min_diff)
            sig_index = sample(range(len(dataset)), max_source)
            data = [dataset[i] for i in sig_index]
            
            signals = []; class_names = []; active_masks = []
            for (audio, class_name, active_mask) in data:
                signals.append(audio)
                class_names.append(class_name)  
                active_masks.append(active_mask)
            signals = self._simulate(signals, room, source_locs)
            if self.HRTF:
                signals = self.hrtf_simulator.apply_HRTF(signals, doas)
            for j, s in enumerate(signals):
                os.makedirs(f"{audio_folder}/{i}", exist_ok=True)
                sf.write(f"{audio_folder}/{i}/{j}.wav", s.T, self.sr)

            # save the frame-level meta data
            meta_file = f"{meta_folder}/{i}.csv"
            label = []
            for k in range(len(class_names)):
                active_mask = active_masks[k]
                active_frames = np.where(active_mask)[0].tolist()
                for frame in active_frames:
                    label.append([frame, class_names[k], k, doas[k][0], doas[k][1], ranges[k]])
            df = pd.DataFrame(label, columns=['frame', 'class', 'source', 'azimuth', 'elevation', 'distance'])
            df = df.sort_values(by=['frame'])
            df.to_csv(meta_file, index=False)

   