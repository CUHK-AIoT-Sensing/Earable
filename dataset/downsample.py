import scipy.io.wavfile as wavfile
import librosa
import os
from tqdm import tqdm
import numpy as np

new_rate = 16000
dataset = 'knowles/'
speakers = os.listdir(dataset)
for speaker in speakers:
    for date in os.listdir(dataset + speaker):
        print(speaker, date)
        for wav in tqdm(os.listdir(dataset + speaker + "/" + date)):
            if wav[-3:] != "wav":
                continue
            path = dataset + speaker + "/" + date + "/" + wav
            data, sample_rate = librosa.load(path, mono=False, sr=None)
            vibration, audio = data[0], data[1]
            vibration *= 4 # amplify the vibration
            data = np.stack((audio, vibration)) # swap channel
            samples = round(len(data) * float(new_rate) / sample_rate)
            new_data = librosa.resample(data, orig_sr=sample_rate, target_sr=new_rate, scale=True)
            wavfile.write(dataset + speaker + "/" + date + "/" + wav, new_rate, new_data.T)