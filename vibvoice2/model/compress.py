import time
import soundfile as sf
import scipy
import torchaudio
import torch
import os
import numpy as np
bit_rate = '16k'
def flac_codec(data, scale=8):
    tmp_file = '/tmp/' + str(time.time())
    b, a = scipy.signal.butter(4, 1/scale)
    data_mic = data[0]
    sf.write(tmp_file + '_mic.flac', data_mic, 16000)
    data_imu = scipy.signal.filtfilt(b, a, data[1], axis=0)[::scale]
    sf.write(tmp_file + '_imu.flac', data_imu, 16000//scale)

    ori_size = data.nbytes
    compress_size = os.path.getsize(tmp_file + '_mic.flac') + os.path.getsize(tmp_file + '_imu.flac')

    data_mic = torchaudio.load(tmp_file + '_mic.flac')[0]
    data_imu = torchaudio.load(tmp_file + '_imu.flac', )[0]
    data_imu = torch.nn.functional.interpolate(data_imu.unsqueeze(1), scale_factor=scale, mode='linear', align_corners=False).squeeze(1)
    data = torch.cat([data_mic, data_imu], dim=0)
    os.remove(tmp_file + '_mic.flac')
    os.remove(tmp_file + '_imu.flac')
    return data, compress_size/ori_size
def mp3_codec(data):
    # MP3 lossy compression
    tmp_file = '/tmp/' + str(time.time()) + '.mp3'
    scaled_data = np.int16(data.numpy() * 32767)
    scipy.io.wavfile.write(tmp_file.replace('.mp3', '.wav'), 16000, scaled_data.T)
    os.system('ffmpeg -i '+ tmp_file.replace('.mp3', '.wav') +' -vn -ar 16000 -ac 2 -b:a ' + bit_rate + ' ' + tmp_file + ' -loglevel quiet')
    ori_size = os.path.getsize(tmp_file.replace('.mp3', '.wav'))
    compress_size = os.path.getsize(tmp_file)
    data, _ = torchaudio.load(tmp_file)
    os.remove(tmp_file)   
    os.remove(tmp_file.replace('.mp3', '.wav'))
    return data, compress_size/ori_size