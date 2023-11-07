import soundfile as sf
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
def SI_SDR(reference, estimation, sr=16000):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    References
        SDR– Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy

    projection = optimal_scaling * reference

    noise = estimation - projection
    ratio = np.sum(projection ** 2, axis=-1) / (np.sum(noise ** 2, axis=-1) + 1e-6)
    return 10 * np.log10(ratio)
class Flac():
    def __init__(self,):
        pass
    def load(self, file):
        self.data, self.samplerate = sf.read(file)
        self.file = file
        self.output_file = file.replace('.wav', '.flac')
        self.channel_1_file = self.output_file.replace('.flac', '_channel_1.flac')
        self.channel_2_file = self.output_file.replace('.flac', '_channel_2.flac')
    def latency(self, encode, decode):
        t_start = time.time()
        for i in range(100):
            encode()
        t_end = time.time()
        print('encode latency: ', (t_end - t_start)/100)

        t_start = time.time()
        for i in range(100):
            decode()
        t_end = time.time()
        print('decode latency: ', (t_end - t_start)/100)

    def encode(self):
        sf.write(self.output_file, self.data, self.samplerate)
    def decode(self):
        self.compressed_data, self.samplerate = sf.read(self.output_file)
    def compare(self):
        size_wav = os.stat(self.file).st_size
        size_flac = os.stat(self.output_file).st_size
        print('compression ratio: ', size_flac/size_wav *100 , '%')
        sisnr = SI_SDR(self.data.T, self.compressed_data.T)
        print('SI-SDR: ', sisnr)

    def encode_stereo_decorrelation(self, scale=8):
        mid = (self.data[:, 0] + self.data[:, 1])/2
        side = self.data[:, 0]
        sf.write(self.channel_1_file, mid, self.samplerate)
        sf.write(self.channel_2_file, side, self.samplerate)
    def decode_stereo_decorrelation(self, scale=8):
        self.data_1, self.samplerate = sf.read(self.channel_1_file)
        self.data_2, self.samplerate = sf.read(self.channel_2_file)
        left = self.data_2
        right = (self.data_1 * 2 - self.data_2)
        self.compressed_data = np.stack([left, right], axis=1)
    def encode_stereo(self, scale=8):
        b, a = scipy.signal.butter(4, 1/scale)
        data_1 = scipy.signal.filtfilt(b, a, self.data[:, 0], axis=0)[::scale]
        data_2 = self.data[:, 1]
        sf.write(self.channel_1_file, data_1, self.samplerate//scale)
        sf.write(self.channel_2_file, data_2, self.samplerate)
    def decode_stereo(self, scale=8):
        self.data_1, self.samplerate = sf.read(self.channel_1_file)
        self.data_2, self.samplerate = sf.read(self.channel_2_file)
        self.data_1 = np.interp(np.arange(len(self.data_1) * scale), np.arange(0, len(self.data_1)*scale, scale), self.data_1)

        self.compressed_data = np.stack([self.data_1, self.data_2], axis=1)
    def compare_stereo(self):
        size_wav = os.stat(self.file).st_size
        size_flac = os.stat(self.channel_1_file).st_size + os.stat(self.channel_2_file).st_size
        print('compression ratio: ', size_flac/size_wav *100 , '%')
        sisnr = SI_SDR(self.data.T, self.compressed_data.T)
        print('SI-SDR: ', sisnr)
        fig, axs = plt.subplots(2)
        axs[0].plot(self.data[:,0])
        axs[1].plot(self.compressed_data[:,0])
        axs[1].plot(self.data[:,0] - self.compressed_data[:,0])

        plt.savefig('compare.png')
    
flac = Flac()
flac.load('example.wav')
flac.encode()
flac.decode()
flac.latency(flac.encode, flac.decode)
flac.compare()

flac.encode_stereo_decorrelation()
flac.decode_stereo_decorrelation()
flac.latency(flac.encode_stereo_decorrelation, flac.decode_stereo_decorrelation)
flac.compare_stereo()

flac.encode_stereo()
flac.decode_stereo()
flac.latency(flac.encode_stereo, flac.decode_stereo)

flac.compare_stereo()

