import torch
from dataset import EMSBDataset, ABCSDataset, VoiceBankDataset
import model
import argparse
from tqdm import tqdm
import numpy as np
import json
import os
import datetime

def test_epoch(model, dataset, BATCH_SIZE, device):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    Metric = []
    model.eval()
    with torch.no_grad():
        for sample in pbar:
            noisy = sample['noisy'].to(device).squeeze(1); clean = sample['clean'].to(device).squeeze(1); noise = sample['noise'].to(device).squeeze(1)
            noisy_mag = torch.stft(noisy, 640, 320, 640, window=torch.hann_window(640).to(device), return_complex=True).abs()
            clean = torch.nn.functional.pad(clean, (320, 320))
            noise = torch.nn.functional.pad(noise, (320, 320))
            window = torch.hann_window(640).to(device)
            snr = torch.sum((window * clean.unfold(1, 640, 320))**2, dim=-1) / (torch.sum((window * noise.unfold(1, 640, 320))**2, dim=-1) + 1e-8)
            snr = 10 * torch.log(snr)
            snr = torch.clip(snr, min=-10, max=35)
            snr = (snr + 10) / 45 
            snr_predict = model(noisy_mag)
            loss = torch.nn.functional.l1_loss(snr_predict, snr)
            Metric.append(loss.item() * 45 - 10)  
    avg_metric = np.mean(Metric)
    print(avg_metric)
    return avg_metric
def train_epoch(model, train_loader, optimizer, device):
    pbar = tqdm(train_loader)
    Metric = []
    model.train()
    for sample in pbar:
        noisy = sample['noisy'].to(device).squeeze(1); clean = sample['clean'].to(device).squeeze(1); noise = sample['noise'].to(device).squeeze(1)
        noisy_mag = torch.stft(noisy, 640, 320, 640, window=torch.hann_window(640).to(device), return_complex=True).abs()
        clean = torch.nn.functional.pad(clean, (320, 320))
        noise = torch.nn.functional.pad(noise, (320, 320))
        window = torch.hann_window(640).to(device)
        snr = torch.sum((window * clean.unfold(1, 640, 320))**2, dim=-1) / (torch.sum((window * noise.unfold(1, 640, 320))**2, dim=-1) + 1e-8)
        snr = 10 * torch.log(snr)
        snr = torch.clip(snr, min=-10, max=35)
        snr = (snr + 10) / 45 
        optimizer.zero_grad()
        snr_predict = model(noisy_mag)
        loss = torch.nn.functional.mse_loss(snr_predict, snr)
        loss.backward()
        optimizer.step()
        pbar.set_description("loss: %.3f" % (loss.item()))
        Metric.append(loss.item())  
    return np.mean(Metric)
def train(dataset, EPOCH, lr, BATCH_SIZE, model,):
    train_dataset, test_dataset = dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, pin_memory=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    save_dir = 'checkpoints/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    os.mkdir(save_dir)
    loss_best = 1000
    ckpt_best = model.state_dict()
    if checkpoint is not None:
        print('first test the initial checkpoint')
        avg_metric = test_epoch(model, test_dataset, BATCH_SIZE, device)
    for e in range(EPOCH):
        mean_lost = train_epoch(model, train_loader, optimizer, device)
        avg_metric = test_epoch(model, test_dataset, BATCH_SIZE, device)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = avg_metric
            torch.save(ckpt_best, save_dir + args.model + '_' + args.dataset + '_' + str(e) + '_' + str(metric_best) + '.pth')
    torch.save(ckpt_best, save_dir + 'best.pth')
    print('best performance is', metric_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", default=False, required=False)
    parser.add_argument('--model', action="store", type=str, default='SNR_Predictor', required=False, help='choose the model')
    parser.add_argument('--dataset', '-d', action="store", type=str, default='ABCS', required=False, help='choose the mode')

    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # model_name = args.model
    model = getattr(model, args.model)().to(device)
    # model = torch.nn.DataParallel(model)
    rir = 'json/rir.json'
    BATCH_SIZE = 16
    lr = 0.00001
    EPOCH = 5
    checkpoint = None
    # checkpoint = '20230918-190354/best.pth'
    noises = [
              'json/ASR_aishell-dev.json',
              'json/other_DEMAND.json',
              # 'json/other_freesound.json'
              ]
    noise_file = []
    for noise in noises:
        noise_file += json.load(open(noise, 'r'))

    mode = 'PN'
        
    if args.dataset == 'EMSB':
        dataset = [EMSBDataset('json/EMSB.json', noise=noise_file, ratio=0.8, mono=True, mode=mode), EMSBDataset('json/EMSB.json', noise=noise_file, ratio=-0.2, mono=True)]
    elif args.dataset == 'ABCS':
        dataset = [ABCSDataset('json/ABCS_train.json', noise=noise_file, mode=mode), 
                   ABCSDataset('json/ABCS_dev.json', noise=noise_file)]
    elif args.dataset == 'VoiceBank':
        dataset = [VoiceBankDataset('json/voicebank_clean_trainset_wav.json', noise=noise_file, mode=mode), VoiceBankDataset('json/voicebank_clean_testset_wav.json', noise=noise_file)]
    else:
        raise ValueError('dataset not found')
    if checkpoint is not None:
        ckpt = torch.load('checkpoints/' + checkpoint)
        model.load_state_dict(ckpt, strict=True)
    if args.train:
        train(dataset, EPOCH, lr, BATCH_SIZE, model)
    else:
        avg_metric = test_epoch(model, dataset[-1], BATCH_SIZE, device)

      