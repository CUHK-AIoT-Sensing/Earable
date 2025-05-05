import torch
from dataset import EMSBDataset, ABCSDataset, VoiceBankDataset, V2SDataset
import model
import argparse
from feature import ASR, predict_sisnr
import json
import os
import datetime
import numpy as np
def train(dataset, EPOCH, BATCH_SIZE, model, optimizer, epoch, loss_best, discriminator = None, optimizer_disc = None):
    train_dataset, test_dataset = dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, pin_memory=True)
    if checkpoint is None:
        save_dir = 'checkpoints/' + args.model + '_' + args.arch + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
        os.mkdir(save_dir)
    else:
        save_dir = 'checkpoints/' + checkpoint + '/'

    for e in range(epoch+1, EPOCH):
        mean_lost = train_epoch(model, train_loader, optimizer, device, discriminator, optimizer_disc)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            optimizer_best = optimizer.state_dict()
            loss_best = mean_lost
            torch.save({
            'epoch': e,
            'model_state_dict': ckpt_best,
            'optimizer_state_dict': optimizer_best,
            'loss': loss_best,
            },  save_dir + str(loss_best) + '.pt')
    test_epoch(model, test_dataset, BATCH_SIZE, device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", default=False, required=False)
    parser.add_argument('--adversarial', action="store_true", default=False, required=False)
    parser.add_argument('--arch', action="store", type=str, default='VibVoice', required=False, help='choose the model')
    parser.add_argument('--model', action="store", type=str, default='masker', choices=['masker', 'vocoder', 'filter'], required=False, help='choose the model')
    parser.add_argument('--tta', action="store_true", default=False, required=False)
    parser.add_argument('--save', action="store_true", default=False, required=False)
    parser.add_argument('--dataset', '-d', action="store", type=str, default='ABCS', required=False, help='choose the mode')

    args = parser.parse_args()
    torch.cuda.set_device(0)
    import importlib
    trainer_module = importlib.import_module('trainer.' + args.model)
    train_epoch = trainer_module.train_epoch
    train_epoch_tta = trainer_module.train_epoch_tta
    test_epoch = trainer_module.test_epoch
    test_epoch_save = trainer_module.test_epoch_save

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    rir = 'json/rir.json'
    BATCH_SIZE = 16
    lr = 0.0001
    EPOCH = 20
    se_model = getattr(model, args.model)(args.arch).to(device)
    optimizer = torch.optim.Adam(params=se_model.parameters(), lr=lr)
    if args.adversarial:
        discriminator = model.Discriminator().to(device)
        optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=2 * lr)
    else:
        discriminator = None
        optimizer_disc = None
    checkpoint = None
    # checkpoint = 'generator_20240118-143700'
    # checkpoint = 'masker_20240123-140050'
    # checkpoint = 'filter_20240122-132406'
    # 
    checkpoint = 'masker_VibVoice_Lite_20240124-171926'
    # checkpoint = 'Baseline_20240122-170839'

    # checkpoint = 'vocoder_VibVoice_20240118-143700'
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
    elif args.dataset == 'V2S':
        dataset = [V2SDataset('json/V2S.json', length=5), V2SDataset('json/V2S.json', length=None)]
        dir = '../V2S/'
        output_dir = '../V2S_tmp/'
    if checkpoint is not None:
        list_ckpt = os.listdir('checkpoints/' + checkpoint)
        ckpts = [float(ckpt[:-3]) for ckpt in list_ckpt]
        ckpt_name = 'checkpoints/' + checkpoint + '/' + list_ckpt[np.argmin(ckpts)]
        ckpt = torch.load(ckpt_name)
        se_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        loss_best = ckpt['loss']
        print('load checkpoint:', ckpt_name)
    else:
        loss_best = 1000
        epoch = -1
    if args.train:
        train(dataset, EPOCH, BATCH_SIZE, se_model, optimizer, epoch, loss_best, discriminator, optimizer_disc)
    else:
        if args.dataset == 'V2S':
            import pickle
            if args.tta: # only tta at real world dataset
                mean_lost = train_epoch_tta(se_model, dataset[-1], dir, output_dir, device='cuda', method='BN_adapt')
            elif args.save:
                test_epoch_save(se_model, dataset[-1], dir, output_dir, device)
            
            result_dict, bad_cases = ASR(output_dir)
            with open('saved_dict.pkl', 'wb') as f:
                pickle.dump(result_dict, f)
            with open('bad_cases.pkl', 'wb') as f:
                pickle.dump(bad_cases, f)
            #result_dict = pickle.load(open('saved_dict.pkl', 'rb')) 
            print(result_dict)
            #snr_dict = predict_sisnr(dir, output_dir)
            # print(snr_dict)
            # result_dict = list(result_dict.values())
            # print(np.mean(result_dict), np.std(result_dict))
        else:
            test_epoch(se_model, dataset[-1], BATCH_SIZE, device)
            # for snr in [[-5, 0], [0, 5], [5, 10]]:
            #     test_dataset = ABCSDataset('json/ABCS_dev.json', noise=noise_file, snr=snr)
            #     print('snr', snr)
            #     test_epoch(se_model, test_dataset, BATCH_SIZE, device)
            # for noise in ['json/other_freesound.json', 'json/ASR_aishell-dev.json', 'json/other_DEMAND.json']:
            #     noise_file = json.load(open(noise, 'r'))
            #     test_dataset = ABCSDataset('json/ABCS_dev.json', noise=noise_file)
            #     print('noise', noise)
            #     test_epoch(se_model, test_dataset, BATCH_SIZE, device)


      
