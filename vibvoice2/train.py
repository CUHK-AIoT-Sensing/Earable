import torch
from dataset import EMSBDataset, ABCSDataset, VoiceBankDataset
import model
import argparse
from helper import train_epoch, test_epoch
import json
import os
import datetime
def train(dataset, EPOCH, lr, BATCH_SIZE, model,):
    train_dataset, test_dataset = dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, pin_memory=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_best = 1000
    ckpt_best = model.state_dict()
    if checkpoint is None:
        save_dir = 'checkpoints/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
        os.mkdir(save_dir)
    else:
        save_dir = 'checkpoints/' + checkpoint + '/'
    for e in range(EPOCH):
        mean_lost = train_epoch(model, train_loader, optimizer, device)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            torch.save(ckpt_best, save_dir + args.model + '_' + args.dataset + '_' + str(e) + '_' + str(loss_best) + '.pth')
    test_epoch(model, test_dataset, BATCH_SIZE, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", default=False, required=False)
    parser.add_argument('--model', action="store", type=str, default='DPCRN', required=False, help='choose the model')
    parser.add_argument('--dataset', '-d', action="store", type=str, default='ABCS', required=False, help='choose the mode')

    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # model_name = args.model
    se_model = getattr(model, args.model)().to(device)
    # model = torch.nn.DataParallel(model)
    rir = 'json/rir.json'
    BATCH_SIZE = 16
    lr = 0.0001
    EPOCH = 20
    checkpoint = None
    os.makedirs('tmp', exist_ok=True)
    # checkpoint = '20230918-190354/best.pth'
    noises = [
              'json/ASR_aishell-dev.json',
              'json/other_DEMAND.json',
              ]
    noise_file = []
    for noise in noises:
        noise_file += json.load(open(noise, 'r'))

    if args.dataset == 'EMSB':
        dataset = [EMSBDataset('json/EMSB.json', noise=noise_file, ratio=0.8, mono=True), 
                   EMSBDataset('json/EMSB.json', noise=noise_file, ratio=-0.2, mono=True)]
    elif args.dataset == 'ABCS':
        dataset = [ABCSDataset('json/ABCS_train.json', noise=noise_file), 
                   ABCSDataset('json/ABCS_dev.json', noise=noise_file)]
    elif args.dataset == 'VoiceBank':
        dataset = [VoiceBankDataset('json/voicebank_clean_trainset_wav.json', noise=noise_file), 
                   VoiceBankDataset('json/voicebank_clean_testset_wav.json', noise=noise_file)]
    else:
        raise ValueError('dataset not found')
    if checkpoint is not None:
        ckpt = torch.load('checkpoints/' + checkpoint)
        se_model.load_state_dict(ckpt, strict=True)

    if args.train:
        train(dataset, EPOCH, lr, BATCH_SIZE, se_model)
    else:
        test_epoch(se_model, dataset[-1], 4, device)

      