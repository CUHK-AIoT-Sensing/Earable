'''
This script prepares the real inference and output saving
It also contains the unsupervised learning
'''
import torch
from dataset import V2SDataset, ABCSDataset
import model
import argparse
from helper import unsupervised_train_epoch, test_epoch_save, AlterNet_E, AlterNet_M
import json
import datetime
import os
def train(dataset, EPOCH, lr, BATCH_SIZE, model, teacher_model):
    train_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, pin_memory=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    # optimizer1 = torch.optim.Adam(params=teacher_model.parameters(), lr=lr)
    # optimizer2 = torch.optim.Adam(params=model.parameters(), lr=lr)
    save_dir = 'checkpoints/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    os.mkdir(save_dir)
    loss_best = 1000
    ckpt_best = model.state_dict()
    for e in range(EPOCH):
        mean_lost = unsupervised_train_epoch(model, train_loader, optimizer, teacher_model, device='cuda')
        # if e < EPOCH//2:
        #     mean_lost = AlterNet_E(model, train_loader, optimizer1, device='cuda')
        # else:
        #     mean_lost = AlterNet_M(model, train_loader, optimizer2, device='cuda')
        print('epoch', e, 'loss', mean_lost)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
    torch.save(ckpt_best, save_dir + 'best.pth')
    print('best loss is', loss_best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", default=False, required=False)
    parser.add_argument('--model', action="store", type=str, default='DPCRN', required=False, help='choose the model')
    parser.add_argument('--dataset', action="store", type=str, default='V2S', required=False, help='choose the model')

    args = parser.parse_args()
    torch.cuda.set_device(1)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    se_model = getattr(model, args.model)().to(device)
    teacher_model = getattr(model, args.model)().to(device)
    # teacher_model = getattr(model, 'SNR_Predictor')().to(device)
    # teacher_model = getattr(model, 'DPCRN_basic')().to(device)
    rir = 'json/rir.json'
    BATCH_SIZE = 16
    lr = 0.00001
    EPOCH = 20
    checkpoint = None
    if checkpoint is not None:
        ckpt = torch.load('checkpoints/' + checkpoint)
        se_model.load_state_dict(ckpt, strict=True)
        teacher_model.load_state_dict(ckpt, strict=True)
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
        dataset = V2SDataset('json/V2S.json', length=5)
        dir = '../V2S/'
        output_dir = '../V2S_tmp/'
    if args.train:
        train(dataset, EPOCH, lr, BATCH_SIZE, se_model, teacher_model)
    # test_dataset = V2SDataset('json/V2S.json', length=None)
    # test_epoch_save(se_model, test_dataset, BATCH_SIZE, dir, output_dir, device)

      