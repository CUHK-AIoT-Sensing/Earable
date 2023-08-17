from dataset import EMSBDataset, ABCSDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import VAD
from feature import stft, istft
import torchmetrics


def train(dataset, EPOCH, lr, BATCH_SIZE, model,):
    if isinstance(dataset, list):
        # with pre-defined train/ test
        train_dataset, test_dataset = dataset
    else:
        # without pre-defined train/ test
        length = len(dataset)
        test_size = min(int(0.2 * length), 2000)
        train_size = length - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    loss_best = 100
    ckpt_best = model.state_dict()
    loss_function = torch.nn.BCELoss()
    metic = torchmetrics.classification.BinaryAccuracy()
    for e in range(EPOCH):
        Loss_list = []
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for sample in train_loader:
                mag, phase, real, imag = stft(sample['imu'], 640, 320, 640)
                mag = mag[:, :, :33, :]
                vad = sample['vad']
                # loss = torch.zeros(1)
                loss = loss_function(model(mag), vad.reshape(-1, 1))
                loss.backward()
                optimizer.step()
                t.set_description('Epoch %i' % e)
                t.set_postfix(loss=loss.item())
                t.update(1)
        mean_lost = np.mean(Loss_list)
        scheduler.step()
        accuracy = []
        model.eval()
        with torch.no_grad():
            for sample in tqdm(test_loader):
                mag, phase, real, imag = stft(sample['imu'], 640, 320, 640)
                mag = mag[:, :, :33, :]
                vad = sample['vad']
                accuracy.append(((model(mag) > 0.5) == vad.reshape(-1, 1)).sum()/(BATCH_SIZE * vad.shape[-1]))
                # accuracy.append(metic(model(mag), vad.reshape(-1, 1)).item())
                # print(binary_acc, accuracy[-1])
        avg_metric = np.mean(accuracy)
        print(avg_metric)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = avg_metric
    torch.save(ckpt_best, 'pretrain/' + str(metric_best) + '.pth')
    return ckpt_best, metric_best


if __name__ == "__main__":
    # dataset = EMSBDataset('json/EMSB.json', ratio=1, mono=True)
    dataset = [ABCSDataset('json/ABCS_train.json'), ABCSDataset('json/ABCS_dev.json')]
    train(dataset, 20, 1e-3, 32, VAD())
    # train_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=2, batch_size=2, shuffle=True)
    # for data in train_loader:
    #     vad = data['vad'][0].T.repeat_interleave(320)
    #     plt.subplot(1, 2, 1)
    #     plt.plot(data['clean'][0].T)
    #     plt.plot(vad * data['clean'][0].T.max())
    #     plt.subplot(1, 2, 2)
    #     plt.plot(data['imu'][0].T)
    #     plt.plot(vad * data['imu'][0].T.max())
    #     plt.savefig('test.png')
    #     break