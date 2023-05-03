import torch.nn as nn
import torch
import time
from torch.utils.mobile_optimizer import optimize_for_mobile
import os
import numpy as np

freq_bin_high = 33

def synthetic(clean, transfer_function, N):
    time_bin = clean.shape[-1]
    index = np.random.randint(0, N)
    f = transfer_function[index, :, 0]
    f = f / np.max(f)
    v = transfer_function[index, :, 1] / np.max(f)
    response = np.tile(np.expand_dims(f, axis=1), (1, time_bin))
    for j in range(time_bin):
        response[:, j] += np.random.normal(0, v, (freq_bin_high))
    response = torch.from_numpy(response).to(clean.device)
    acc = clean[..., :freq_bin_high, :] * response
    return acc

class IMU_branch(nn.Module):
    def __init__(self):
        super(IMU_branch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(2, 1), dilation=(2, 1)),
            #nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(2, 1), dilation=(2, 1)),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(2, 1), dilation=(2, 1)),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            #nn.ConvTranspose2d(64, 64, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ConvTranspose2d(32, 32, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ConvTranspose2d(16, 16, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.final = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        # down-sample
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_mid = self.conv4(x)
        # up-sample with supervisi
        x = self.conv5(x_mid)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.final(x)
        return x_mid, x

class Audio_branch(nn.Module):
    def __init__(self):
        super(Audio_branch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=4, dilation=2),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=4, dilation=2),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2, dilation=1),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, dilation=1),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return [x1, x2, x3, x4, x5]

class Residual_Block(nn.Module):
    def __init__(self, in_channels):
        super(Residual_Block, self).__init__()
        self.r1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=5, padding=2, dilation=1),
            nn.ConvTranspose2d(128, 128, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.r2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=5, padding=2, dilation=1),
            nn.ConvTranspose2d(64, 64, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.r3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=5, padding=4, dilation=2),
            nn.ConvTranspose2d(32, 32, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.r4 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=4, dilation=2),
            nn.ConvTranspose2d(16, 16, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ConvTranspose2d(16, 1, kernel_size=(2, 1), stride=(2, 1)),
            nn.Sigmoid()
            )
    def forward(self, acc, audios):

        [x1, x2, x3, x4, x5] = audios
        x = torch.cat([acc, x5], dim=1)
        x = torch.cat([self.r1(x), x4], dim=1)
        x = torch.cat([self.r2(x), x3], dim=1)
        x = torch.cat([self.r3(x), x2], dim=1)
        x = torch.cat([self.r4(x), x1], dim=1)
        x = self.final(x)
        return x

class vibvoice(nn.Module):
    def __init__(self):
        super(vibvoice, self).__init__()
        self.IMU_branch = IMU_branch()
        self.Audio_branch = Audio_branch()
        self.Residual_block = Residual_Block(384)

        self.transfer_function = np.load('function_pool.npy')
        self.length_transfer_function = self.transfer_function.shape[0]
    def norm(self, x):
        mu = torch.mean(x, dim=list(range(1, x.dim())), keepdim=True)
        normed = x / (mu + 1e-5)
        return normed
    def forward(self, noisy, acc=None):
        # Preprocessing
        if acc == None:
            acc = synthetic(noisy, self.transfer_function, self.length_transfer_function)
        else:
            batch = acc.shape[0]
            acc = acc.to(noisy.device).reshape(batch*3, -1)
            acc = torch.abs(torch.stft(acc, 64, 32, 64, window=torch.hann_window(64, device=noisy.device), return_complex=True))
            acc = torch.norm(acc.reshape(batch, 3, 33, -1), dim=1)
        noisy = torch.unsqueeze(noisy[:, 1:257, 1:], 1)
        acc = torch.unsqueeze(acc[:, 1:, 1:], 1)
        acc_mid, acc_output = self.IMU_branch(acc)
        mask = self.Residual_block(acc_mid, self.Audio_branch(noisy))
        clean = mask * noisy
        return clean, acc_output

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

def model_save(model):
    model.eval()
    x = torch.rand((1, 1, 32, 250))
    noise = torch.rand((1, 1, 256, 250))
    scripted_module = torch.jit.trace(model, [x, noise])
    #scripted_module.save("inference.pt")
    optimized_scripted_module = optimize_for_mobile(scripted_module)
    optimized_scripted_module._save_for_lite_interpreter("vibvoice.ptl")
def model_speed(model, input):
    t_start = time.time()
    step = 100
    with torch.no_grad():
        for i in range(step):
            model(*input)
    return (time.time() - t_start)/step
if __name__ == "__main__":

    #acc = torch.rand(1, 1, 32, 250)
    audio = torch.rand(1, 321, 251)
    model = vibvoice()
    tf = model.transfer_function
    for f in tf:
        if (np.isnan(f)).any():
            print('find!')
    audio, acc = model(audio)

    # size_all_mb = model_size(model)
    # print('model size: {:.3f}MB'.format(size_all_mb))

    #latency = model_speed(model, [audio])
    # print('model latency: {:.3f}S'.format(latency))

    #model_save(model)
