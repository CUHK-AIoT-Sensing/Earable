from dataset import EMSBDataset, ABCSDataset
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import VAD_CRN
from feature import stft, istft
import argparse
import sklearn.metrics

import warnings
warnings.filterwarnings("ignore")

def train(dataset, EPOCH, lr, BATCH_SIZE, model,):
    train_dataset, test_dataset = dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    loss_best = 100
    ckpt_best = model.state_dict()
    loss_function = torch.nn.BCELoss()
    mel = torchaudio.transforms.MelScale(64, n_stft=321)
    for e in range(EPOCH):
        Loss_list = []
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for sample in train_loader:
                mag, phase, real, imag = stft(sample['imu'], 640, 320, 640)
                mag = mel(mag)
                vad = sample['vad']
                loss = loss_function(model(mag.to(device)), vad.to(device))
                loss.backward()
                optimizer.step()
                Loss_list.append(loss.item())
                t.set_description('Epoch %i' % e)
                t.set_postfix(loss=np.mean(Loss_list))
                t.update(1)
        mean_lost = np.mean(Loss_list)
        scheduler.step()
        predictions = []; target_label = []
        model.eval()
        with torch.no_grad():
            for sample in tqdm(test_loader):
                mag, phase, real, imag = stft(sample['imu'], 640, 320, 640)
                mag = mel(mag)
                vad = sample['vad']
                predictions.append(model(mag.to(device)).cpu().numpy())
                target_label.append(vad.numpy())
        predictions = np.concatenate(predictions)
        target_label = np.concatenate(target_label).reshape(-1, 1)

        auc = sklearn.metrics.roc_auc_score(target_label, predictions)
        acc = sklearn.metrics.accuracy_score(target_label, predictions > 0.5)
        print(auc, acc)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = auc
    torch.save(ckpt_best, 'checkpoints/vad_' + str(metric_best) + '.pth')
    print('best performance is', metric_best)
    return ckpt_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', action="store", type=int, default=2, required=False, help='choose the mode')
    parser.add_argument('--dataset', '-d', action="store", type=str, default='ABCS', required=False, help='choose the mode')

    args = parser.parse_args()
    if args.dataset == 'EMSB':
        dataset = [EMSBDataset('json/EMSB.json', ratio=0.8, mono=True), EMSBDataset('json/EMSB.json', ratio=-0.2, mono=True)]
    elif args.dataset == 'ABCS':
        dataset = [ABCSDataset('json/ABCS_train.json'), ABCSDataset('json/ABCS_dev.json')]
    else:
        raise ValueError('dataset not found')
    print('train dataset length:', len(dataset[0]), 'test dataset length:', len(dataset[1]))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode == 0:
        train_loader = torch.utils.data.DataLoader(dataset=dataset[1], num_workers=2, batch_size=2, shuffle=True)
        ratio = 0
        for data in train_loader:
            ratio += data['vad'].mean()
        print("VAD ratio:", ratio.item()/len(train_loader))
    elif args.mode == 1:
        train_loader = torch.utils.data.DataLoader(dataset=dataset[1], num_workers=2, batch_size=2, shuffle=True)
        for i, data in enumerate(train_loader):
            vad = data['vad'][0].T.repeat_interleave(320)
            fig = plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(data['clean'][0].T)
            plt.plot(vad * data['clean'][0].T.max())
            plt.subplot(1, 2, 2)
            plt.plot(data['imu'][0].T)
            plt.plot(vad * data['imu'][0].T.max())
            plt.savefig('fig/test_' + str(i) + '.png')
            if i >= 10:
                break
    elif args.mode == 2:
        model = VAD_CRN().to(device)
        train(dataset, 5, 1e-4, 16, model)
    else: # non-deep learning method
        train_dataset, test_dataset = dataset
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=False, pin_memory=False)
        accuracy = []
        for sample in tqdm(test_loader):
            imu = sample['imu'].numpy()[0, 0]
            vad_pred = VAD(imu, sr=16000, nFFT=640, win_length=0.04, hop_length=0.02, theshold=0.6)
            vad = sample['vad'].numpy()
            accuracy.append((vad_pred == vad.reshape(-1, 1)).sum()/(1 * vad.shape[-1]))
        avg_metric = np.mean(accuracy)
        print(avg_metric)
      
        
        