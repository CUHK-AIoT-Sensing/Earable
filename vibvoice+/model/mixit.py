'''
["Audio signal enhancement with learning from positive and unlabelled data"](https://arxiv.org/abs/2210.15143)
'''
import torch
import torch.nn as nn
import torch
from .dpcrn import DPCRN
def MIXIT():
    return DPCRN(last_act=None, last_channel=3)
    # return MIXT_uni()
class MIXT_uni(nn.Module):
    def __init__(self, blocks=2, channels=8, droprate=0.2, fcblocks=0, method='MixIT'):
        super(MIXT_uni, self).__init__()
        modules = []
        modules.append(torch.nn.Conv2d(1, channels, 3, padding = 'same'))
        modules.append(torch.nn.ReLU())
        if method == 'PN':
            modules.append(torch.nn.BatchNorm2d(channels))
        else:
            modules.append(torch.nn.Dropout2d(droprate))
        modules.append(torch.nn.Conv2d(channels, channels, 3, padding = 'same'))
        modules.append(torch.nn.ReLU())
        if method == 'PN':
            modules.append(torch.nn.BatchNorm2d(channels))
        else:
            modules.append(torch.nn.Dropout2d(droprate))
        for i in range(blocks - fcblocks):
            modules.append(torch.nn.Conv2d(channels * (2 ** i), channels * (2 ** (i + 1)), 3, padding = 'same'))
            modules.append(torch.nn.ReLU())
            if method == 'PN':
                modules.append(torch.nn.BatchNorm2d(channels * (2 ** (i + 1))))
            else:
                modules.append(torch.nn.Dropout2d(droprate))
            modules.append(torch.nn.Conv2d(channels * (2 ** (i + 1)), channels * (2 ** (i + 1)), 3, padding = 'same'))
            modules.append(torch.nn.ReLU())
            if method == 'PN':
                modules.append(torch.nn.BatchNorm2d(channels * (2 ** (i + 1))))
            else:
                modules.append(torch.nn.Dropout2d(droprate))
        for i in range(blocks - fcblocks, blocks):
            modules.append(torch.nn.Conv2d(channels * (2 ** i), channels * (2 ** (i + 1)), 1, padding = 'same'))
            modules.append(torch.nn.ReLU())
            if method == 'PN':
                modules.append(torch.nn.BatchNorm2d(channels * (2 ** (i + 1))))
            else:
                modules.append(torch.nn.Dropout2d(droprate))
            modules.append(torch.nn.Conv2d(channels * (2 ** (i + 1)), channels * (2 ** (i + 1)), 1, padding = 'same'))
            modules.append(torch.nn.ReLU())
            if method == 'PN':
                modules.append(torch.nn.BatchNorm2d(channels * (2 ** (i + 1))))
            else:
                modules.append(torch.nn.Dropout2d(droprate))

        if method == 'MixIT':
            modules.append(torch.nn.Conv2d(channels * (2 ** blocks), 3, 3, padding = 'same'))
        elif method == 'PN':
            modules.append(torch.nn.Conv2d(channels * (2 ** blocks), 1, 3, padding = 'same'))
        else:
            modules.append(torch.nn.Conv2d(channels * (2 ** blocks), 1, 1, padding = 'same'))
        
        self.network = torch.nn.Sequential(*modules)

    def forward(self, x, acc):
        return self.network(x)