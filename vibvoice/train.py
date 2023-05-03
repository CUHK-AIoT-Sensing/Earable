import os
import time
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
from dataset import NoisyCleanSet
from model import vibvoice, fullsubnet
import numpy as np
from tqdm.auto import tqdm
import argparse
import model_zoo
def parse_sample(sample, text=False):
    if text:
        data, text = sample[:-1], sample[-1]
    else:
        data = sample
        text = None
    if len(data) == 3:
        clean, noise, acc = data
    else:
        clean, noise = data
        acc = None
    return text, clean, noise, acc

def inference(dataset, BATCH_SIZE, model, text=False):
    text_inference = text
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    Metric = []
    model.eval()
    with torch.no_grad():
        for sample in test_loader:
            text, clean, noise, acc = parse_sample(sample, text=text_inference)
            metric = getattr(model_zoo, 'test_' + model_name)(model, acc, noise, clean, device, text)
            Metric.append(metric)
    avg_metric = np.mean(np.concatenate(Metric, axis=0), axis=0)
    return avg_metric

def train(dataset, EPOCH, lr, BATCH_SIZE, model,):
    if isinstance(dataset, list):
        # with pre-defined train/ test
        train_dataset, test_dataset = dataset
    else:
        # without pre-defined train/ test
        length = len(dataset)
        test_size = min(int(0.1 * length), 2000)
        train_size = length - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, pin_memory=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    loss_best = 100
    loss_curve = []
    ckpt_best = model.state_dict()

    for e in range(EPOCH):
        Loss_list = []
        model.train()
        for i, sample in enumerate(tqdm(train_loader)):
            text, clean, noise, acc = parse_sample(sample)
            loss = getattr(model_zoo, 'train_' + model_name)(model, acc, noise, clean, optimizer, device)
            Loss_list.append(loss)
        mean_lost = np.mean(Loss_list)
        loss_curve.append(mean_lost)
        scheduler.step()
        avg_metric = inference(test_dataset, 16, model)
        print(avg_metric)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = avg_metric
            torch.save(ckpt_best, 'pretrain/' + str(metric_best) + '.pth')
    return ckpt_best, loss_curve, metric_best



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action="store", type=int, default=0, required=False,
                        help='mode of the script')
    parser.add_argument('--model', action="store", type=str, default='vibvoice', required=False,
                        help='choose the model')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # select available model from vibvoice, fullsubnet, conformer,
    model_name = args.model
    model = globals()[model_name]().to(device)
    people = ["hou", "1", "2", "3", "4", "5", "6", "7", "8", "yan", "wu", "liang", "shuai", "shi", "he"]
    if args.mode == 0:
        # This script is for model pre-training on LibriSpeech
        BATCH_SIZE = 64
        lr = 0.0001
        EPOCH = 20
        dataset = NoisyCleanSet(['json/librispeech-100.json', 'json/noises.json'], simulation=True,
                                ratio=1, rir=None, dvector=None)
        ckpt_best, loss_curve, metric_best = train(dataset, EPOCH, lr, BATCH_SIZE, model)
        plt.plot(loss_curve)
        plt.savefig('loss.png')
    elif args.mode == 1:
        # evaluation for personalized model
        dvector = None
        rir = None
        text_evaluation = False
        # start checkpoint
        ckpt_mode = 0
        # per-user train or single-train: 0-per_user, 1-single_train, 2-just_test
        train_mode = 1

        if ckpt_mode == 0:
            ckpt_dir = 'pretrain/vibvoice'
            ckpt_name = ckpt_dir + '/' + sorted(os.listdir(ckpt_dir))[-1]
            ckpt_start = torch.load(ckpt_name)
        elif ckpt_mode == 1:
            ckpt_name = 'pretrain/fullsubnet_[ 2.14935723  2.92502818 13.68871124  0.87719504  5.141906  ].pth'
            ckpt_start = torch.load(ckpt_name)
        else:
            ckpt_name = 'pretrain/[ 2.39249867  2.96939845 14.47059834  0.93109596 18.08529382].pth'
            ckpt_start = torch.load(ckpt_name)
        print('loaded checkpoint:', ckpt_name)

        if train_mode == 0:
            ckpt = []
            for p in people:
                model.load_state_dict(ckpt_start)
                p_except = [i for i in people if i != p]
                train_dataset1 = NoisyCleanSet(['json/noise_train_gt.json', 'json/all_noise.json', 'json/noise_train_imu.json'],
                                             person=[p], simulation=True, rir=rir, dvector=dvector)
                train_dataset2 = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'],
                                              person=[p], simulation=True, rir=rir, ratio=0.8, dvector=dvector)
                train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
                pt, _, _ = train(train_dataset, 5, 0.0001, 4, model)
                ckpt.append(pt)
        elif train_mode == 1:
            model.load_state_dict(ckpt_start)
            train_dataset1 = NoisyCleanSet(['json/noise_train_gt.json', 'json/all_noise.json', 'json/noise_train_imu.json'],
                                           person=people, simulation=True, rir=rir, dvector=dvector)
            train_dataset2 = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'],
                                           person=people, simulation=True, rir=rir, ratio=0.8, dvector=dvector)

            positions = ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']
            train_dataset3 = NoisyCleanSet(['json/position_gt.json', 'json/all_noise.json', 'json/position_imu.json'],
                                           simulation=True, person=positions, ratio=0.8,)

            train_dataset = torch.utils.data.ConcatDataset([train_dataset2, train_dataset3])
            ckpt, _, _ = train(train_dataset, 5, 0.0001, 16, model)
        else:
            ckpt = ckpt_start
            model.load_state_dict(ckpt)

        if text_evaluation:
            if isinstance(ckpt, list):
                for pt, p in zip(ckpt, people):
                    model.load_state_dict(ckpt)
                    test_dataset = NoisyCleanSet(['json/noise_gt.json', 'json/noise_wav.json', 'json/noise_imu.json'],
                                                 person=[p], simulation=not text_evaluation, text=text_evaluation, dvector=dvector)
                    avg_metric = inference(test_dataset, 4, model, text=text_evaluation)
                    print(p, avg_metric)
            else:
                for p in people:
                    model.load_state_dict(ckpt)
                    test_dataset = NoisyCleanSet(['json/noise_gt.json', 'json/noise_wav.json', 'json/noise_imu.json'],
                                                 person=[p], simulation=not text_evaluation, text=text_evaluation, dvector=dvector)
                    avg_metric = inference(test_dataset, 4, model, text=text_evaluation)
                    print(p, avg_metric)

            envs = ['airpod', 'freebud', 'galaxy', 'office', 'corridor', 'stair', 'human-corridor', 'human-hall', 'human-outdoor']
            for env in envs:
                test_dataset = NoisyCleanSet(['json/noise_gt.json', 'json/noise_wav.json', 'json/noise_imu.json'],
                                             person=[env], simulation=not text_evaluation, text=text_evaluation, dvector=dvector)
                avg_metric = inference(test_dataset, 4, model, text=text_evaluation)
                print(env, avg_metric)
        else:
            dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'],
                                    person=people, simulation=True, ratio=-0.2, dvector=dvector)
            avg_metric = inference(dataset, 4, model)
            print('overall performance:', avg_metric)
            exit()
            for p in people:
                dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json', 'json/train_imu.json'],
                                        person=[p], simulation=True, ratio=-0.2, dvector=dvector)
                avg_metric = inference(dataset, 4, model)
                print(p, avg_metric)

            for noise in ['background.json', 'dev.json', 'music.json']:
                dataset = NoisyCleanSet(['json/train_gt.json', 'json/' + noise,  'json/train_imu.json'],
                                        person=people, simulation=True, ratio=-0.2, dvector=dvector)
                avg_metric = inference(dataset, 4, model)
                print(noise, avg_metric)

            for level in [10, 5, 1]:
                dataset = NoisyCleanSet(['json/train_gt.json', 'json/all_noise.json',  'json/train_imu.json'], person=people,
                                         simulation=True, snr=[level - 1, level + 1], ratio=-0.2, dvector=dvector)
                avg_metric = inference(dataset, 4, model)
                print(level, avg_metric)

            positions = ['glasses', 'vr-up', 'vr-down', 'headphone-inside', 'headphone-outside', 'cheek', 'temple', 'back', 'nose']
            for p in positions:
                dataset = NoisyCleanSet(['json/position_gt.json', 'json/all_noise.json', 'json/position_imu.json'],
                                        person=[p], simulation=True, ratio=-0.2, dvector=dvector)
                avg_metric = inference(dataset, 4, model)
                print(p, avg_metric)

            for p in ['he', 'hou']:
                test_dataset = NoisyCleanSet(['json/mobile_gt.json', 'json/all_noise.json', 'json/mask_imu.json'],
                                             person=[p], simulation=True, dvector=dvector)
                avg_metric = inference(test_dataset, 4, model)
                print(p, avg_metric)

            test_dataset = NoisyCleanSet(['json/mask_gt.json', 'json/all_noise.json', 'json/mask_imu.json'],
                                         person=['he'], simulation=True, dvector=dvector)
            avg_metric = inference(test_dataset, 4, model)
            print('mask', avg_metric)


    else:
        # investigate the performance of FullSubnet
        checkpoint = torch.load("fullsubnet_best_model_58epochs.tar")
        print('loading pre-trained FullSubNet (SOTA)', checkpoint['best_score'])
        model.load_state_dict(checkpoint['model'])

        dataset = NoisyCleanSet(['json/DNSclean.json', 'json/DNSnoisy.json'],
                                simulation=False, ratio=1)
        avg_metric = inference(dataset, 4, model)
        print(avg_metric)

        # dataset = NoisyCleanSet(['json/dev.json', 'json/cv.json'],
        #                            simulation=True, ratio=0.1, rir='json/rir.json')
        # avg_metric = inference(dataset, 4, model)
        # print(avg_metric)
