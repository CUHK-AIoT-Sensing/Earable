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
    if args.method == 'AlterNet':
        optimizer1 = torch.optim.Adam(params=teacher_model.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(params=model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
  
    save_dir = 'checkpoints/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    os.mkdir(save_dir)
    loss_best = 1000
    ckpt_best = model.state_dict()
    for e in range(EPOCH):
        if args.method == 'AlterNet':
            if e < EPOCH//2:
                mean_lost = AlterNet_E(model, train_loader, optimizer1, device='cuda')
            else:
                mean_lost = AlterNet_M(model, train_loader, optimizer2, device='cuda')
        else:
            mean_lost = unsupervised_train_epoch(model, train_loader, optimizer, teacher_model, device='cuda', method=args.method)
       
        print('epoch', e, 'loss', mean_lost)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
    torch.save(ckpt_best, save_dir +  args.method + '_' + args.model + '_' + args.dataset + 'best.pth')
    print('best loss is', loss_best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", default=False, required=False)
    parser.add_argument('--model', action="store", type=str, default='DPCRN', required=False, help='choose the model')
    parser.add_argument('--dataset', action="store", type=str, default='V2S', required=False, help='choose the dataset')
    parser.add_argument('--method', action="store", type=str, default='AlterNet', choices=['AlterNet', 'PSE_DP', 'RemixIT'], required=False, help='choose the method')

    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    se_model = getattr(model, args.model)().to(device)
    
    rir = 'json/rir.json'
    BATCH_SIZE = 16
    lr = 0.00001
    EPOCH = 20
    checkpoint = '20231228-111830'
    if checkpoint is not None:
        list_ckpt = os.listdir('checkpoints/' + checkpoint)
        list_ckpt.sort()
        ckpt_name = 'checkpoints/' + checkpoint + '/' + list_ckpt[-1]
        print('load checkpoint:', ckpt_name)
        ckpt = torch.load(ckpt_name)
        se_model.load_state_dict(ckpt, strict=True)
    # alternet: small translator from IMU to audio, PSE_DP: SNR predictor, RemixIT: same as se_model
    if args.method == 'AlterNet':
        teacher_model = getattr(model, 'DPCRN_basic')().to(device) # no pre-trained
    elif args.method == 'PSE_DP':
        teacher_model = None
    else:
        teacher_model = getattr(model, args.model)().to(device)
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

    test_dataset = V2SDataset('json/V2S.json', length=None)
    test_epoch_save(se_model, test_dataset, dir, output_dir, device)

      