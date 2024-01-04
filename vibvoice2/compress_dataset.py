import json
import os
from tqdm import tqdm
from multiprocessing import Pool
import tqdm
import time

def _foo(d):
    f, length = d
    f_new = f.replace('wav', 'mp3').replace('Audio', 'mp3_' + bit_rate)
    os.makedirs(os.path.dirname(f_new), exist_ok=True)
    if 'k' in bit_rate:
        os.system('ffmpeg -i '+ f +' -vn -ar 16000 -ac 2 -b:a ' + bit_rate + ' ' + f_new + ' -loglevel quiet -y')
    else:
        os.system('ffmpeg -i '+ f +' -vn -ar 16000 -ac 2 -q:a ' + bit_rate + ' ' + f_new + ' -loglevel quiet -y')
    ori_size = os.path.getsize(f)
    compress_size = os.path.getsize(f_new)
    compress_ratio = compress_size/ori_size
    return compress_ratio 

if __name__ == '__main__':
    data = './json/ABCS_train.json'  
    # bit_rate = '96k'
    bit_rate = '0'
    with open(data, 'r') as f:
        data = json.load(f)
        left = []
        for speaker in data.keys():
            left += data[speaker]
    # left = left[:50]
    with Pool(8) as p:
        r = list(tqdm.tqdm(p.imap(_foo, left), total=len(left)))
    print(len(r), sum(r)/len(r))







