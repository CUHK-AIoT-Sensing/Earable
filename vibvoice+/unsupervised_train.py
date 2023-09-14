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
import numpy as np
import json

def remix(predict, noise):
    permuted_t_est_noise = noise[torch.randperm(noise.shape[0])]
    bootstrapped_mix = predict + permuted_t_est_noise
    return bootstrapped_mix
def clip(sample, length=16000*5):
    for key in ['predict', 'noise', 'imu']:
        data = sample[key]
        if data.shape[-1] > length:
            remain_length = data.shape[-1] % length
            data = torch.nn.functional.pad(data, (0, length - remain_length, 0, 0), 'constant')
            sample[key] = data.reshape(-1, 1, length)
        else:
            data = torch.nn.functional.pad(data, (0, length - data.shape[-1], 0, 0), 'constant')
            sample[key] = data.reshape(-1, 1, length)
    #    active = sample['vad']
    return sample
def bootstrapped_training(dataset, BATCH_SIZE, model, teacher_model):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=True)
    teacher_model.eval()
    model.train()
    for name,child in (model.named_children()):
        if name.find('BatchNorm') != -1:
            for param in child.parameters():
                param.requires_grad = False 

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001)
    Loss_list = []
    with tqdm(total=len(test_loader) // BATCH_SIZE) as t:
        for i, sample in enumerate(test_loader):
            with torch.no_grad():
                clean, predict = getattr(helper, 'test_' + model_name)(teacher_model, sample, device)
                clean = torch.from_numpy(clean); predict = torch.from_numpy(predict)
                noise = clean - predict
                sample['predict'] = predict
                sample['noise'] = noise
            if i % BATCH_SIZE == 0:
                if i != 0:
                    batch_sample['noisy'] = remix(batch_sample['predict'], batch_sample['noise'])
                    batch_sample['clean'] = batch_sample['predict']
                    loss = getattr(helper, 'train_' + model_name)(model, batch_sample, optimizer, device)
                    Loss_list.append(loss)
                    t.set_postfix(loss=np.mean(Loss_list))
                    t.update(1)
                batch_sample = clip(sample)
            else:
                sample = clip(sample)
                for key in ['predict', 'noise', 'imu']:
                    batch_sample[key] = torch.cat([batch_sample[key], sample[key]], dim=0)

    # model.eval()
    # with torch.no_grad():
    #     for i, sample in enumerate(tqdm(test_loader)):
    #         _, predict = getattr(helper, 'test_' + model_name)(model, sample, device)
    #         for j in range(1):
    #             fname = sample['file'][j]
    #             fname = fname.replace(dir, output_dir)
    #             wavfile.write(fname, 16000, predict[j])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action="store", type=str, default='DPCRN', required=False, help='choose the model')

    args = parser.parse_args()
    torch.cuda.set_device(1)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model_name = args.model
    teacher_model = globals()[model_name]().to(device)
    model = globals()[model_name]().to(device)
    checkpoint = '20230903-145505/best.pth'
    ckpt = torch.load('checkpoints/' + checkpoint)
    teacher_model.load_state_dict(ckpt, strict=True)
    model.load_state_dict(ckpt, strict=True)

    dir = '../V2S/'
    output_dir = '../V2S_tmp/'
    BATCH_SIZE = 4
    with open('json/V2S.json', 'r') as f:
        data = json.load(f)
        data_list = []
        for speaker in data.keys():
            for date in data[speaker]:
                print(speaker, date)
                dataset = V2SDataset(data[speaker][date], mode='WILD')
                bootstrapped_training(dataset, BATCH_SIZE, model, teacher_model)


      