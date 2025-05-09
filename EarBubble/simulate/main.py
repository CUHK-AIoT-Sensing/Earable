import os
from simulator import simulator
from audio_dataset import dataset_parser
import numpy as np
import json
from wasn import simulation_to_wasn
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TIMIT', choices=['TIMIT', 'VCTK', 'NIGENS', 'AudioSet', 'FSD50K'])
    parser.add_argument('--save_folder', type=str, required=False, default='../../dataset')
    parser.add_argument('--device', type=str, default='earphone', choices=['earphone', 'smartglass'])
    parser.add_argument('--mode', type=str, default='event', choices=['event', 'soundscape'])
    parser.add_argument('--num_data', type=int, default=500)
    parser.add_argument('--sr', type=int, default=44100)
    config = {'max_source': 1, 'num_user': 2, 'num_source': 100,}

    args = parser.parse_args()
    train_dataset, test_dataset = dataset_parser(args.dataset, '../../dataset/audio', args.sr)  
    dataset_folder = args.save_folder + '/simulation/{}_{}_{}_{}/'.format(args.device, args.dataset, args.mode, config['max_source'])

    train_folder = os.path.join(dataset_folder, 'train'); test_folder = os.path.join(dataset_folder, 'test')
    os.makedirs(train_folder, exist_ok=True); os.makedirs(test_folder, exist_ok=True) 
    ism_simulator = simulator(args.device, args.sr)
    TRAIN_NUM = args.num_data; TEST_NUM = None if args.num_data is None else args.num_data//5

    ism_simulator.sound_event(train_folder, train_dataset, num_data=TRAIN_NUM, config=config)
    ism_simulator.sound_event(test_folder, test_dataset, num_data=TEST_NUM, config=config)