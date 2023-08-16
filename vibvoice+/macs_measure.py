from model import *
import torch
import argparse
from ptflops import get_model_complexity_info
import numpy as np
import time

def latency_measure(model, data, device='cuda'):
    # causal inference is not suitable for synchronize timing
    # INIT LOGGERS
    model.to(device)
    data = {k: v.to(device) for k, v in data.items()}
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(**data)
    # MEASURE PERFORMANCE
    t_start = time.time()
    with torch.no_grad():
        for rep in range(repetitions):
            # starter.record()
            _ = model(**data)
            # ender.record()
            # WAIT FOR GPU SYNC
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            # timings[rep] = curr_time
    mean_syn = (time.time() - t_start) / repetitions
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    print(device, 'latency:', mean_syn, 'RTF:', (data['x'].shape[-1]/50)/mean_syn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', action="store", type=str, default='CRN', required=False, help='choose the model')
    args = parser.parse_args()

    if args.model == 'CRN':
        model = CRN()
        input_contructor = lambda x: {'x':torch.rand(1, 1, 321, 50), 'acc':torch.rand(1, 1, 33, 50)}
    elif args.model == 'vibvoice':
        pass
    else: # vad
        model = VAD()
        input_contructor = lambda x: {'acc':torch.rand(1, 1, 33, 250)}
    model.eval()
    macs, params = get_model_complexity_info(model, input_res=(1, 321, 150), as_strings=True, input_constructor=input_contructor,print_per_layer_stat=False , verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    latency_measure(model, input_contructor(0), device='cpu')


    # Export the model
    # torch.onnx.export(model,               # model being run
    #                 input_contructor(0),                         # model input (or a tuple for multiple inputs)
    #                 args.model + ".onnx",   # where to save the model (can be a file or file-like object)
    #                 export_params=True,        # store the trained parameter weights inside the model file
    #                 opset_version=10,          # the ONNX version to export the model to
    #                 do_constant_folding=True,  # whether to execute constant folding for optimization
    #                 input_names = ['input'],   # the model's input names
    #                 output_names = ['output'], # the model's output names
    #                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #                                 'output' : {0 : 'batch_size'}})