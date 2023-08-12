from model import *
import torch
import argparse
from ptflops import get_model_complexity_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', action="store", type=str, default='VAD', required=False, help='choose the model')
    args = parser.parse_args()

    if args.model == 'CRN':
        model = CRN()
        input_contructor = lambda x: {'x':torch.rand(1, 1, 321, 150), 'acc':torch.rand(1, 1, 33, 150)}
    elif args.model == 'vibvoice':
        pass
    else: # vad
        model = VAD()
        input_contructor = lambda x: {'acc':torch.rand(1, 1, 33, 150)}

    macs, params = get_model_complexity_info(model, input_res=(1, 321, 150), as_strings=True, input_constructor=input_contructor,print_per_layer_stat=True, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))