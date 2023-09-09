import scipy.io.wavfile as wavfile
import librosa
import os
from tqdm import tqdm

new_rate = 16000
dataset = 'V2S/'
for speaker in os.listdir(dataset):
    for date in os.listdir(dataset + speaker):
        print(speaker, date)
        for wav in tqdm(os.listdir(dataset + speaker + "/" + date)):
            if wav[-3:] != "wav":
                continue
            path = dataset + speaker + "/" + date + "/" + wav
            data, sample_rate = librosa.load(path, mono=False, sr=None)
            samples = round(len(data) * float(new_rate) / sample_rate)
            new_data = librosa.resample(data, orig_sr=sample_rate, target_sr=new_rate, scale=True)
            wavfile.write(dataset + speaker + "/" + date + "/" + wav, new_rate, new_data.T)