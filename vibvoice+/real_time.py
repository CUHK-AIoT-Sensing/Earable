from model import DPCRN
import torch
import time
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action="store", type=str, default='Conv', required=False, help='choose the model')
    args = parser.parse_args()

    dummy_input1 = torch.randn(1, 1, 321, 251)
    dummy_input2 = torch.randn(1, 1, 321, 251)
    net = DPCRN()
    net.eval()
    real_time_output = []
    t_start = time.time()
    for i in range(dummy_input1.shape[-1]):
        real_time_input1 = dummy_input1[:, :, :, i:i+1]
        real_time_input2 = dummy_input2[:, :, :, i:i+1]
        real_time_output.append(net.forward_causal(real_time_input1, real_time_input2))
    t_end = time.time()
    inference_latency_per_frame = round((t_end - t_start)/dummy_input1.shape[-1],4)
    load_latency_per_frame = 1/50
    overall_latency_per_frame = inference_latency_per_frame + load_latency_per_frame
    print(inference_latency_per_frame, load_latency_per_frame)
    print('Latency:', overall_latency_per_frame, 'RTF:', inference_latency_per_frame/load_latency_per_frame)

    # measure whether real-time is different from full output
    real_time_output = torch.cat(real_time_output, dim=-1)
    full_output = net(dummy_input1, dummy_input2)
    error = torch.mean(torch.abs((full_output - real_time_output)/full_output))
    print('Real-time difference ' + args.model, error.item())