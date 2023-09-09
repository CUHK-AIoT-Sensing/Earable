'''
1. EMSB Chinese, TWO ears, each file contains AC + BC, 16kHz
2. ABCS, already split into train/dev/test, headset, one file contains AC + BC, 16kHz
3. V2S+
4. VoiceBank
5. Others (Aishell, LibriSpeech)
'''
import torchaudio
import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', action="store", type=str, default='EMSB', required=False)
    args = parser.parse_args()
    print('start processing')
    if args.dataset == 'EMSB':
        sample_rate = 16000
        dict = {}
        directory = '../' + args.dataset
        g = os.walk(directory)
        count = 0; timer = 0
        data = {'left': [], 'right': []}
        for path, dir_list, file_list in g:
            if len(file_list) == 2: # left + right
                left = [os.path.join(path, file_list[0]), torchaudio.info(os.path.join(path, file_list[0])).num_frames]
                right = [os.path.join(path, file_list[1]), torchaudio.info(os.path.join(path, file_list[1])).num_frames]
                data['left'].append(left)
                data['right'].append(right)
                timer += min(left[1], right[1])/16000
                count += 1
        print('dataset summary', args.dataset, 'we have recording:', count, 'whole duration (hour):', timer/3600)
        json.dump(data, open('json/' + args.dataset + '.json', 'w'), indent=4)
    elif args.dataset == 'ABCS':
        sample_rate = 16000
        splits = ['train', 'dev', 'test']
        for split in splits:
            data = {}
            directory = '../' + args.dataset + '/Audio/' + split
            timer = 0
            for speaker in os.listdir(directory):
                path = directory + '/' + speaker
                data[speaker] = []
                for f in os.listdir(path):
                    headset = [os.path.join(path, f), torchaudio.info(os.path.join(path, f)).num_frames]
                    data[speaker].append(headset)
                    timer += headset[1]/16000
            print('dataset summary', args.dataset, split, 'we have recording:', len(os.listdir(directory)), 'whole duration (hour):', timer/3600)
            json.dump(data, open('json/' + args.dataset + '_' + split + '.json', 'w'), indent=4)
    elif args.dataset == 'voicebank':
        sample_rate = 16000
        splits = ['clean_trainset_wav', 'clean_testset_wav',]
        for split in splits:
            data = {}
            directory = '../other/' + args.dataset + '/' + split
            timer = 0
            data = []
            for f in os.listdir(directory):
                headset = [os.path.join(directory, f), torchaudio.info(os.path.join(directory, f)).num_frames]
                data.append(headset)
                timer += headset[1]/16000
            print('dataset summary', args.dataset, split, 'we have recording:', len(os.listdir(directory)), 'whole duration (hour):', timer/3600)
            json.dump(data, open('json/' + args.dataset + '_' + split + '.json', 'w'), indent=4)
    elif args.dataset == 'V2S':
        sample_rate = 16000
        data = {}
        directory = '../' + args.dataset 
        speakers = os.listdir(directory)
        timer = 0
        for speaker in speakers:
            path = directory + '/' + speaker
            data[speaker] = []
            for date in os.listdir(path):
                date_path = path + '/' + date
                for f in os.listdir(date_path):
                    try:
                        headset = [os.path.join(date_path, f), torchaudio.info(os.path.join(date_path, f)).num_frames]
                        data[speaker].append(headset)
                        timer += headset[1]/16000
                    except:
                        pass
        print('dataset summary', args.dataset, 'we have speaker:', len(os.listdir(directory)), 'whole duration (hour):', timer/3600)
        json.dump(data, open('json/' + args.dataset + '.json', 'w'), indent=4)
    elif args.dataset == 'other':
        sample_rate = 16000
        splits = ['background', 'librispeech-dev', 'rir', 'DEMAND', 'aishell-dev']
        for split in splits:
            data = []
            directory = '../' + args.dataset + '/' + split
            g = os.walk(directory)
            count = 0; timer = 0
            for path, dir_list, file_list in g:
                if len(file_list) > 0:
                    for f in os.listdir(path):
                        try:
                            audio = [os.path.join(path, f), torchaudio.info(os.path.join(path, f)).num_frames]
                            data.append(audio)
                            timer += audio[1]/16000
                            count += 1
                        except:
                            pass
            print('dataset summary', args.dataset, split, 'we have recording:', count, 'whole duration (hour):', timer/3600)
            json.dump(data, open('json/' + args.dataset + '_' + split + '.json', 'w'), indent=4)
    elif args.dataset == 'ASR':
        sample_rate = 16000
        datasets = {'LibriSpeech': ['dev-clean', 'dev-other'], 'aishell': ['aishell-dev', 'aishell-test', 'aishell-train']}
        for dataset in datasets:
            for split in datasets[dataset]:
                data = []
                directory = '../' + args.dataset + '/' + dataset + '/' + split
                g = os.walk(directory)
                count = 0; timer = 0
                for path, dir_list, file_list in g:
                    if len(file_list) > 0:
                        for f in os.listdir(path):
                            try:
                                audio = [os.path.join(path, f), torchaudio.info(os.path.join(path, f)).num_frames]
                                data.append(audio)
                                timer += audio[1]/16000
                                count += 1
                            except:
                                pass
                print('dataset summary', args.dataset, split, 'we have recording:', count, 'whole duration (hour):', timer/3600)
                json.dump(data, open('json/' + args.dataset + '_' + split + '.json', 'w'), indent=4)

