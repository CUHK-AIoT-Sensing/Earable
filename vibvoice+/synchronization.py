import os

import matplotlib.pyplot as plt
import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf
import argparse
import scipy.interpolate as interpolate

def calibrate(file, T, shift):
    data = np.loadtxt(file)
    timestamp = data[:, -1]
    # data = data[:, :3] / 2 ** 14
    # data = data - np.mean(data, axis=0)
    data = data[:, :3]
    f = interpolate.interp1d(timestamp - timestamp[0], data, axis=0, kind='nearest')
    t = min((timestamp[-1] - timestamp[0]), T)
    num_sample = int(T * rate_imu)
    data = np.zeros((num_sample, 3))
    xnew = np.linspace(0, t, num_sample)
    data[shift:num_sample, :] = f(xnew)[:-shift, :]
    return data
def directory_decompose(files):
    import itertools
    dict = {}
    for k, v in itertools.groupby(files, key=lambda x: x.replace(' ', '_').split('_')[0]):
        dict[k] = list(v)
    return dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', action="store", type=int, default=0, required=False)
    args = parser.parse_args()
    # synchronize airpods and microphone
    source = 'dataset/raw'
    target = 'dataset/our'
    # for p in ['he', 'yan', 'hou', 'shi', 'shuai', 'wu', 'liang', "1", "2", "3", "4", "5", "6", "7", "8", 'jiang', '9']:
    #for p in ['airpod', 'freebud', 'galaxy','vr-down', 'vr-up', 'headphone-inside', 'headphone-outside', 'glasses', 'cheek', 'temple', 'back', 'nose']:
    for p in ['human-corridor', 'human-outdoor', 'human-hall']:
        print(p)
        g = os.walk(os.path.join(source, p))
        for path, dir_list, file_list in g:
            N = len(file_list)
            if N > 0:
                name = path.split('\\')[-1]
                if name in ['noise', 'clean', 'mobile']:
                    T = 5
                else:
                    T = 30
                dict = directory_decompose(file_list)
                imu1 = dict['bmiacc1']
                imu2 = dict['bmiacc2']
                gt = dict['mic']
                android = False
                if '新录音' in dict:
                    dual = True
                    wav = dict['新录音']
                elif '20220209' in dict:
                    # Special case for the android recording
                    dual = True
                    android = True
                    wav = dict['20220209']
                else:
                    dual = False
                folder = target + path[11:]
                for i in range(len(imu1)):
                    time1 = float(imu1[i].split('_')[1][:-4])
                    time2 = float(imu2[i].split('_')[1][:-4])
                    time_mic = float(gt[i].split('_')[1][:-4])
                    shift1 = int((time1 - time_mic) * 1600)
                    shift2 = int((time2 - time_mic) * 1600)

                    data1 = calibrate(os.path.join(path, imu1[i]), T, shift1)
                    data2 = calibrate(os.path.join(path, imu2[i]), T, shift2)
                    mic = librosa.load(os.path.join(path, gt[i]), sr=16000)[0]
                    # fig, axs = plt.subplots(3)
                    # axs[0].plot(data1)
                    # axs[1].plot(data2)
                    # axs[2].plot(mic)
                    # plt.show()

                    if dual:
                        airpods = librosa.load(os.path.join(path, wav[i]), sr=16000)[0]
                        shift = np.argmax(signal.correlate(mic, airpods)) - np.shape(mic)
                        airpods = np.roll(airpods, shift)[-T * 16000:]
                        airpods = np.pad(airpods, (0, T * 16000 - len(airpods)))

                        f2 = wav[i][:-4] + '.wav'
                        if android:
                            f2 = 'new' + wav[i][:-4] + '.wav'
                        sf.write(os.path.join(folder, f2), airpods, 16000)
                    f1 = gt[i][:-4] + '.wav'
                    sf.write(os.path.join(folder, f1), mic, 16000)
                    np.savetxt(os.path.join(folder, imu1[i]), data1, fmt='%1.2f')
                    np.savetxt(os.path.join(folder, imu2[i]), data2, fmt='%1.2f')




