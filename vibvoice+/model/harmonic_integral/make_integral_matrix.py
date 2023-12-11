# coding: utf-8
# Author：WangTianRui
# Date ：2021-07-13 15:09
import numpy as np
import librosa as lib
import matplotlib.pyplot as plt


def make_integral_matrix():
    n_fft = 320
    freq_res = 0.1
    f_max = 8000
    max_harmonic = int(420 /freq_res)
    min_harmonic = int(60 / freq_res) 
    factor = np.zeros((max_harmonic, n_fft+1))
    harmonic_loc = np.zeros((max_harmonic, n_fft+1))
    for f in range(min_harmonic, max_harmonic):
        last_loc = 0
        for k in range(1, int(f_max / freq_res // f) + 1):
            compress_freq_loc = int(f * k / (f_max / freq_res) * n_fft)
            value = 1 / np.sqrt(k)
            factor[f, compress_freq_loc] += value
            harmonic_loc[f, compress_freq_loc] = 1.0
            # 谷结构建模
            if compress_freq_loc - last_loc > 1:
                if (last_loc + compress_freq_loc) % 2 != 0:
                    first_loc = int((last_loc + compress_freq_loc) // 2)
                    second_loc = first_loc + 1
                    factor[f, first_loc] += -0.5 * value
                    factor[f, second_loc] += -0.5 * value
                else:
                    loc = int((last_loc + compress_freq_loc) // 2)
                    factor[f, loc] += -1 * value
            # elif compress_freq_loc - last_loc == 1:
            else:
                factor[f, compress_freq_loc] = factor[f, compress_freq_loc] - value * 0.5
                factor[f, last_loc] = factor[f, last_loc] - value * 0.5
            last_loc = compress_freq_loc

    np.save("harmonic_integrate_matrix", factor)
    np.save("harmonic_loc", harmonic_loc)
    plt.imshow(factor, aspect='auto', origin='lower')
    plt.savefig("harmonic_integrate_matrix.png")
    plt.imshow(harmonic_loc, aspect='auto', origin='lower')
    plt.savefig("harmonic_loc.png")
    return factor


if __name__ == '__main__':
    make_integral_matrix()