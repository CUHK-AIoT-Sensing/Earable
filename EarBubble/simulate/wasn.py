'''
Convert the simulated room to WASN-format
'''

import os
import json
import numpy as np
from tqdm import tqdm



def add_virtual_estimation(mic_centers, mic_orientations, source_idx):
    '''
    mic_centers: [N, 3], the center of the microphones
    mic_orientations: [N], the orientation of the microphones

    source_idx: [M] the index of the sources, can be larger than 1 

    output:
    doas: [M, N], the doas between the microphones and the source
    sn_distances: [M, N], the distances between the microphones and the source
    '''
    n_mics = len(mic_centers); n_sources = len(source_idx)
    doas = np.zeros((n_sources, n_mics)); sn_distances = np.zeros((n_sources, n_mics))
    for m in range(n_sources):
        for n in range(n_mics):
            mic_center = mic_centers[n]; mic_orientation = mic_orientations[n]; source = mic_centers[source_idx[m]]
            # get the distance between the mic and the source
            sn_distance = np.linalg.norm(mic_center - source)
            # get the doa between the mic and the source
            if sn_distance == 0:
                doa = 0
            else:
                doa = np.arctan2(source[1] - mic_center[1], source[0] - mic_center[0])
                doa = (doa - mic_orientation) % (2 * np.pi)
            doas[m, n] = doa
            sn_distances[m, n] = sn_distance
    return doas, sn_distances

def add_virtual_estimate_infrastructure(speaker_positions, source_idx, node_positions, node_orientation):
    n_mics = len(node_positions); n_sources = len(source_idx)
    doas = np.zeros((n_sources, n_mics)); sn_distances = np.zeros((n_sources, n_mics))
    for m in range(n_sources):
        for n in range(n_mics):
            mic_center = node_positions[n]; mic_orientation = node_orientation[n]; source = speaker_positions[source_idx[m]]
            # get the distance between the mic and the source
            sn_distance = np.linalg.norm(mic_center - source)
            # get the doa between the mic and the source
            if sn_distance == 0:
                doa = 0
            else:
                doa = np.arctan2(source[1] - mic_center[1], source[0] - mic_center[0])
                doa = (doa - mic_orientation) % (2 * np.pi)
            doas[m, n] = doa
            sn_distances[m, n] = sn_distance
    return doas, sn_distances

def simulation_to_wasn(dataset_dir='../dataset/smartglass/TIMIT/train'):
    '''
    dataset_dir: the directory of the simulated dataset
    output_file: the output file of the WASN-format
    '''
    output_data = {}
    dataset_dir = f'{dataset_dir}/meta'
    rooms = os.listdir(dataset_dir); rooms.sort()
    for room in tqdm(rooms):
        output_data[room] = {'geometry': {'node_positions': None, 'node_orientations': None, 'node_translations': [], 'node_rotations': []}, 
                            'observations': {'estimates': {'doas': [], 'sn_distances': [], 'translations': [], 'rotations': []},}}
        room_dir = os.path.join(dataset_dir, room)
        sources = os.listdir(room_dir); sources.sort()
        for i, source in enumerate(sources):
            source_file = os.path.join(room_dir, source)
            with open(source_file, 'r') as f:
                data = json.load(f)
            mic_centers = data['mic_centers']; mic_orientations = data['mic_orientations']; source_idx = data['source_idx']
            mic_centers = np.array(mic_centers); mic_orientations = np.array(mic_orientations)
            # degree to radian
            mic_orientations = np.array(mic_orientations) * np.pi / 180

            if i == 0:
                output_data[room]['geometry']['node_positions'] = mic_centers.T
                output_data[room]['geometry']['node_orientations'] = mic_orientations
            doas, sn_distances = add_virtual_estimation(mic_centers, mic_orientations, source_idx)

            if i == 0:
                translations = np.zeros_like(mic_centers)
                rotations = np.zeros_like(mic_orientations)
            else:
                translations = mic_centers - previous_mic_center
                rotations = mic_orientations - previous_mic_orientation
            previous_mic_center = mic_centers; previous_mic_orientation = mic_orientations

            output_data[room]['geometry']['node_translations'].append(translations)
            output_data[room]['geometry']['node_rotations'].append(rotations)

            output_data[room]['observations']['estimates']['doas'].append(doas)
            output_data[room]['observations']['estimates']['sn_distances'].append(sn_distances)
            output_data[room]['observations']['estimates']['translations'].append(translations)
            output_data[room]['observations']['estimates']['rotations'].append(rotations) 
        # convert to numpy array    
        output_data[room]['geometry']['node_translations'] = np.array(output_data[room]['geometry']['node_translations'])
        output_data[room]['geometry']['node_rotations'] = np.array(output_data[room]['geometry']['node_rotations'])
        output_data[room]['observations']['estimates']['doas'] = np.array(output_data[room]['observations']['estimates']['doas'])
        output_data[room]['observations']['estimates']['sn_distances'] = np.array(output_data[room]['observations']['estimates']['sn_distances'])
        output_data[room]['observations']['estimates']['translations'] = np.array(output_data[room]['observations']['estimates']['translations'])
        output_data[room]['observations']['estimates']['rotations'] = np.array(output_data[room]['observations']['estimates']['rotations'])
    return output_data

def simulation_to_infrastructure(dataset_dir='../dataset/smartglass/TIMIT/train', node_positions=[], node_orientations=[]):
    output_data = {}
    dataset_dir = f'{dataset_dir}/meta'
    rooms = os.listdir(dataset_dir); rooms.sort()
    for room in tqdm(rooms):
        output_data[room] = {'geometry': {'node_positions': None, 'node_orientations': None, 'node_translations': [], 'node_rotations': []}, 
                            'observations': {'estimates': {'doas': [], 'sn_distances': [], 'translations': [], 'rotations': []},}}
        room_dir = os.path.join(dataset_dir, room)
        sources = os.listdir(room_dir); sources.sort()
        # we have fixed node 
        output_data[room]['geometry']['node_positions'] = node_positions.T
        output_data[room]['geometry']['node_orientations'] = node_orientations
        for i, source in enumerate(sources):
            source_file = os.path.join(room_dir, source)
            with open(source_file, 'r') as f:
                data = json.load(f)
            mic_centers = data['mic_centers']; mic_orientations = data['mic_orientations']; source_idx = data['source_idx']
            mic_centers = np.array(mic_centers); mic_orientations = np.array(mic_orientations) * np.pi / 180
            doas, sn_distances = add_virtual_estimate_infrastructure(mic_centers, source_idx, node_positions, node_orientations)
            translation = np.zeros_like(node_positions); rotation = np.zeros_like(node_orientations)
            output_data[room]['geometry']['node_translations'].append(translation)
            output_data[room]['geometry']['node_rotations'].append(rotation)

            output_data[room]['observations']['estimates']['doas'].append(doas)
            output_data[room]['observations']['estimates']['sn_distances'].append(sn_distances)
            output_data[room]['observations']['estimates']['translations'].append(translation)
            output_data[room]['observations']['estimates']['rotations'].append(rotation)
    return output_data

def wasn_to_slam(wasn_data, num_users=5):
    '''
    Convert the WASN-format to SLAM-format
    split it for each user
    '''
    output_data = {}
    for user_idx in range(num_users):
        slam_data = {}
        for room, data in wasn_data.items():
            slam_data[room] = {}
            slam_data[room]['geometry'] = {}; slam_data[room]['observations'] = {'estimates': {}}
            slam_data[room]['geometry']['node_positions'] = data['geometry']['node_positions'][:, user_idx]
            slam_data[room]['geometry']['node_orientations'] = data['geometry']['node_orientations'][user_idx]
            slam_data[room]['geometry']['node_translations'] = data['geometry']['node_translations'][:, user_idx]
            slam_data[room]['geometry']['node_rotations'] = data['geometry']['node_rotations'][:, user_idx]

            slam_data[room]['observations']['estimates']['doas'] = data['observations']['estimates']['doas'][..., user_idx]
            slam_data[room]['observations']['estimates']['sn_distances'] = data['observations']['estimates']['sn_distances'][..., user_idx]
            slam_data[room]['observations']['estimates']['translations'] = data['observations']['estimates']['translations'][:, user_idx]
            slam_data[room]['observations']['estimates']['rotations'] = data['observations']['estimates']['rotations'][:, user_idx]
        output_data[str(user_idx)] = slam_data
    return output_data

        
        


if __name__ == '__main__':
    # simulation_to_wasn('../../dataset/simulation/conversation_1_5_500/train')
    simulation_to_infrastructure('../../dataset/simulation/conversation_1_5_500/test')