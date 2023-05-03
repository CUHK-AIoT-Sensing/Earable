import torchaudio
import json
import os
import argparse

def load(path, files, audio=False):
    log = []
    if audio:
        for f in files:
            log.append([os.path.join(path, f), torchaudio.info(os.path.join(path, f)).num_frames])
    else:
        for f in files:
            log.append([os.path.join(path, f), len(open(os.path.join(path, f), 'rb').readlines())])
    return log

def update(dict, name, file_list, p, kinds=4):
    file_list = sorted(file_list)
    N = len(file_list)
    N = int(N / kinds)
    imu1 = file_list[: N]
    imu2 = file_list[N: 2 * N]
    gt = file_list[2 * N: 3 * N]
    wav = file_list[3 * N:]
    imu_files = load(path, imu1, audio=False) + load(path, imu2, audio=False)
    gt_files = load(path, gt, audio=True) + load(path, gt, audio=True)
    wav_files = load(path, wav, audio=True) + load(path, wav, audio=True)
    if name in dict:
        if p in dict[name][0]:
            dict[name][0][p] += imu_files
            dict[name][1][p] += gt_files
            dict[name][2][p] += wav_files
        else:
            dict[name][0][p] = imu_files
            dict[name][1][p] = gt_files
            dict[name][2][p] = wav_files
    else:
        dict[name] = [{p: imu_files}, {p: gt_files}, {p: wav_files}]
    return dict
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of processing, 0-audio only, 1-audio+acc')
    parser.add_argument('--data_dir', action="store", type=str, required=True)
    args = parser.parse_args()
    directory = args.data_dir
    if args.mode == 0:
        # For the audio-only dataset
        datasets = ['dev', 'background', 'music', 'librispeech-100', 'rir_fullsubnet/rir']
        print('processing audio dataset')
        for dataset in datasets:
            print('processing' + dataset)
            audio_files = []
            g = os.walk(os.path.join(directory + dataset))
            dataset_name = dataset.split('/')[-1]
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if file_name[-3:] in ['wav', 'lac']:
                        audio_files.append([os.path.join(path, file_name), torchaudio.info(os.path.join(path, file_name)).num_frames])
            json.dump(audio_files, open('json/' + dataset_name + '.json', 'w'), indent=4)
        noises = []
        datasets = ['dev', 'background', 'music']
        for d in datasets:
            with open('json/' + d + '.json', 'r') as f:
                noises += json.load(f)
        json.dump(noises, open('json/' + 'noises.json', 'w'), indent=4)
    else:
        print('processing audio-imu dataset')
        person = os.listdir(directory)
        dict = {}
        for p in person:
            print('processing', p)
            g = os.walk(os.path.join(directory, p))
            for path, dir_list, file_list in g:
                N = len(file_list)
                if N > 0:
                    # Linux
                    name = path.split('/')[-1]
                    if name in ['test', 'mask', 'position', 'stick']:
                        dict = update(dict, name, file_list, p, kinds=3)
                    else:
                        dict = update(dict, name, file_list, p, kinds=4)
        for name in dict:
                json.dump(dict[name][0], open('json/' + name + '_imu.json', 'w'), indent=4)
                json.dump(dict[name][1], open('json/' + name + '_gt.json', 'w'), indent=4)
                json.dump(dict[name][2], open('json/' + name + '_wav.json', 'w'), indent=4)
