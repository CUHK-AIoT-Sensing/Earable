'''
This script prepares the real inference and output saving
It also contains test time adaptation (TTA)
'''
import torch
from dataset import EMSBDataset, ABCSDataset, V2SDataset
from model import *
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import argparse
import helper


def inference(dataset, BATCH_SIZE, model):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            _, predict = getattr(helper, 'test_' + model_name)(model, sample, device)
            for j in range(BATCH_SIZE):
                fname = sample['file'][j]
                fname = fname.replace(dir, output_dir)
                wavfile.write(fname, 16000, predict[j])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action="store", type=str, default='DPCRN', required=False, help='choose the model')

    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # select available model from vibvoice, fullsubnet, conformer,
    model_name = args.model
    model = globals()[model_name]().to(device)
    rir = 'json/rir.json'
    BATCH_SIZE = 1
    noise = None
    dataset = V2SDataset('json/V2S.json', noise=noise, mode='WILD')
    dir = '../V2S/'
    output_dir = '../V2S_tmp/'
    checkpoint = '20230903-145505/best.pth'
    ckpt = torch.load('checkpoints/' + checkpoint)
    model.load_state_dict(ckpt, strict=True)

    inference(dataset, BATCH_SIZE, model)

      