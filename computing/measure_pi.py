import utils
import onnxruntime as ort
import numpy as np
import torch
import time
from identification.vggm import VGGM

def inference_onnx(onnx_model, sample_input):
    ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
    t_start = time.time()
    step = 100
    for i in range(step):
        outputs = ort_session.run(
            None,
            {"input": sample_input.astype(np.float32)},
        )
    fps = (time.time() - t_start) / step
    return fps

def inference_torch(quant_model, sample_input):
    torch.backends.quantized.engine = 'qnnpack'
    model = VGGM(1251)
    model_int8 = utils.model_quantize(model)
    ckpt = torch.load(quant_model)
    model_int8.load_state_dict(ckpt)
    model_int8.eval()
    #model_int8 = torch.jit.script(model_int8)
    t_start = time.time()
    step = 100
    for i in range(step):
        output = model_int8(sample_input)
    fps = (time.time() - t_start) / step
    return fps
if __name__ == "__main__":
    torch.set_num_threads(1)
    sample_input = np.random.random((1, 1, 512, 300))
    print('identification FPS onnx', inference_onnx('identification.onnx', sample_input))

    sample_input = torch.randn([1, 1, 512, 300])
    print('identification FPS quantized torch', inference_torch('identification_quant.pth', sample_input))