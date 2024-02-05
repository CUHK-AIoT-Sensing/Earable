import os
from scipy.io.wavfile import read, write
dataset_split = 'train'
audio_folder = '../ABCS/Audio/' + dataset_split + '/'
script_folder = '../ABCS/script/' + dataset_split + '/'
output_folder = '../ABCS/vocoder/' + 'wavs/'
file_list = []
os.makedirs(output_folder, exist_ok=True)
for script in os.listdir(script_folder):
    if script == 'modified_record':
        continue
    lines = open(script_folder + script).readlines()
    for line in lines:
        line = line.split(' ')
        fname = line[0]
        text = ' '.join(line[1:])
        speaker_folder = fname.split('_')[0]
        sr, data = read(os.path.join(audio_folder, speaker_folder, fname) + '.wav')
        data = data[:, :1]
        write(output_folder + fname + '.wav', sr, data)
        file_list.append(fname + '|' + text)
        # break
with open('../ABCS/vocoder/' + dataset_split + '.txt', 'w') as f:
    f.write(''.join(file_list))