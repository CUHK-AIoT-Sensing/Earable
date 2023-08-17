from model import CRN
import torch

x = torch.randn(1, 1, 321, 51)
acc = torch.randn(1, 1, 33, 51)

model = CRN()
default_output = model(x, acc)

