from model import *
import torch
def constructor_crn(res):
    audio = torch.rand(1, 321, 150)
    acc = torch.rand(1, 33, 150)
    return dict(x=audio, acc=acc)
def constructor_tcnn(res):
    audio = torch.rand(1, 1, 16000)
    return dict(input=audio)
from ptflops import get_model_complexity_info
model = CRN()
# model = TCNN()
macs, params = get_model_complexity_info(model, input_res=(1, 321, 150), as_strings=True, input_constructor=constructor_crn,
                                        print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))