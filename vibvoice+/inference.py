'''
This script prepares the real inference and output saving
It also contains test time adaptation (TTA)
'''
import torch
from dataset import V2SDataset, ABCSDataset
from model import *
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import argparse
import helper
import scipy.stats as stats
import json
import matplotlib.pyplot as plt
import numpy as np

def inference(dataset, BATCH_SIZE, model):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            clean, predict = getattr(helper, 'test_' + model_name)(model, sample, device)
            if len(clean.shape) == 3:
                for j in range(BATCH_SIZE):
                    fname = sample['file'][j]
                    fname = fname.replace(dir, output_dir)[:-4]
                    np.save(fname.replace('Audio', 'Mel') + '.npy', predict[j])
                    if i % 100 == 0:
                        plt.subplot(1, 2, 1)
                        plt.imshow(clean[j])
                        plt.subplot(1, 2, 2)
                        plt.imshow(predict[j])
                        plt.savefig(fname.replace('Audio', 'Figure')+ '.jpg')

            else:
                for j in range(BATCH_SIZE):
                        fname = sample['file'][j]
                        fname = fname.replace(dir, output_dir)
                        wavfile.write(fname, 16000, predict[j])
def selection_score(sample):
    '''
    1. correlation
    2. entropy + similarity, Efficient Test-Time Model Adaptation without Forgetting, ICML'22
    3. MixIT, MixCycle
    4. PU-learning
    '''
    for i in range(sample['noisy'].shape[0]):
        imu = sample['imu'][i]
        noisy = sample['noisy'][i]
        pearsonr = stats.pearsonr(imu[0], noisy[0])[0]
        print(pearsonr)
def test_time_adaptation(dataset, BATCH_SIZE, model):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    model.train() # activate BN
    for i, sample in enumerate(tqdm(test_loader)):
        # selection_score(sample)
        clean, predict = getattr(helper, 'test_' + model_name)(model, sample, device)
        noise = clean - predict
        # for j in range(BATCH_SIZE):
        #     fname = sample['file'][j]
        #     fname = fname.replace(dir, output_dir)
        #     wavfile.write(fname, 16000, predict[j])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action="store", type=str, default='DPCRN', required=False, help='choose the model')
    parser.add_argument('--TTA', action="store_true", default=False, required=False, help='inference or TTA')
    parser.add_argument('--dataset', action="store", type=str, default='V2S', required=False, help='choose the model')

    args = parser.parse_args()
    torch.cuda.set_device(1)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model_name = args.model
    model = globals()[model_name]().to(device)
    rir = 'json/rir.json'
    BATCH_SIZE = 16
  
    if args.dataset == 'ABCS':
        noises = [
              'json/ASR_aishell-dev.json', 'json/other_DEMAND.json',
              ]
        noise_file = []
        for noise in noises:
            noise_file += json.load(open(noise, 'r'))
        dataset = ABCSDataset('json/ABCS_dev.json', noise=noise_file, length=None)
        dir = '../ABCS/'
        output_dir = '../ABCS_tmp/'
    else:
        with open('json/V2S.json', 'r') as f:
            data = json.load(f)
            data_list = []
            for speaker in data.keys():
                for date in data[speaker]:
                    data_list += data[speaker][date]
        dataset = V2SDataset(data_list, mode='WILD')
        dir = '../V2S/'
        output_dir = '../V2S_tmp/'
    # checkpoint = '20230903-145505/best.pth'
    # checkpoint = '20230913-092655/best.pth'
    # checkpoint  = '20230914-072918/best.pth'
    # checkpoint = '20230914-103229/best.pth'
    # checkpoint = '20230915-112539/best.pth'
    # checkpoint = '20230916-212620/best.pth'
    checkpoint = '20230918-190354/best.pth'
    ckpt = torch.load('checkpoints/' + checkpoint)
    model.load_state_dict(ckpt, strict=True)

    if args.TTA:
        test_time_adaptation(dataset, 1, model)
    else:
        inference(dataset, 1, model)

      