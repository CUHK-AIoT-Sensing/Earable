'''
split the spectrum into two parts, and use two DPCRN to process them respectively.
'''
import torch
import torch.nn as nn
from .dpcrn import DPCRN
import time
class SUB_DPCRN(nn.Module):

    def __init__(self, add=True, early_fusion=False):
        super(SUB_DPCRN, self).__init__()
        self.low = DPCRN([16, 32, 64, 128, 256], add, early_fusion, pad_num=1)
        self.high = DPCRN([16, 32, 64], add, early_fusion=True, pad_num=-1)
        self.bound = 65
    def forward(self, x, acc):
        low_x, high_x = x[:, :, :self.bound, :], x[:, :, self.bound:, :]
        low_acc, high_acc = acc[:, :, :self.bound, :], acc[:, :, self.bound:, :] # high-freq acc is useless
        low_x = self.low(low_x, low_acc)

        # high_x = self.high(high_x, high_acc)
        high_x = self.high(high_x, torch.nn.functional.pad(low_x, (0, 0, 0, 321-65-65), 'constant', 0))
        high_x = torch.nn.functional.pad(high_x, (0, 0, 0, 1), 'constant', 0)
        return torch.cat((low_x, high_x), dim=2)

