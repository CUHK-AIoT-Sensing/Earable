from model import *
import torch
import argparse
from ptflops import get_model_complexity_info
import time

def latency_measure(model, data, device='cuda'):
    # causal inference is not suitable for synchronize timing
    # INIT LOGGERS
    model.to(device)
    data = {k: v.to(device) for k, v in data.items()}
    repetitions = 100
    #GPU-WARM-UP
    for _ in range(50):
        _ = model(**data)
    # MEASURE PERFORMANCE
    t_start = time.time()
    with torch.no_grad():
        for rep in range(repetitions):
            _ = model(**data)
            torch.cuda.synchronize()
    mean_syn = (time.time() - t_start) / repetitions
    print(device, 'latency:', mean_syn, 'RTF:', (data['x'].shape[-1]/50)/mean_syn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', action="store", type=str, default='DPCRN', required=False, help='choose the model')
    args = parser.parse_args()
    model = globals()[args.model]()

    input_contructor = lambda x: {'x':torch.rand(1, 1, 321, 50), 'acc':torch.rand(1, 1, 321, 50)}
 
    model.eval()
    macs, params = get_model_complexity_info(model, input_res=(1, 321, 50), as_strings=True, input_constructor=input_contructor,print_per_layer_stat=False , verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    latency_measure(model, input_contructor(0), device='cpu')
