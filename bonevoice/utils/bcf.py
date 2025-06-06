import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os


def estimate_frequency_response(audio, vibration, fs, nperseg=1024):
    """
    Estimate frequency response H(f) with audio as input and vibration as output.
    
    Parameters:
    audio : array_like, input signal (audio)
    vibration : array_like, output signal (vibration)
    fs : float, sampling frequency in Hz
    nperseg : int, length of each segment for CSD and PSD
    
    Returns:
    freqs : array, frequencies
    H : array, frequency response (complex)
    """
    audio = np.asarray(audio)
    vibration = np.asarray(vibration)
    if len(audio) != len(vibration):
        raise ValueError("Audio and vibration signals must have the same length")
    
    freqs, P_yx = signal.csd(vibration, audio, fs=fs, nperseg=nperseg)
    _, P_xx = signal.welch(audio, fs=fs, nperseg=nperseg)
    epsilon = 1e-10
    H = P_yx / (P_xx + epsilon)
    return freqs, H

def apply_frequency_response(audio, H, freqs, fs):
    """
    Apply frequency response H(f) to audio signal to predict vibration.
    
    Parameters:
    audio : array_like, input signal (audio)
    H : array, frequency response (complex)
    freqs : array, frequencies corresponding to H
    fs : float, sampling frequency in Hz
    
    Returns:
    vibration_pred : array, predicted vibration signal
    """
    audio = np.asarray(audio)
    N = len(audio)
    
    # Compute FFT of audio signal
    audio_fft = fft(audio)
    freqs_full = np.fft.fftfreq(N, 1/fs)
    
    # Interpolate H(f) to match full FFT frequencies
    H_full = np.zeros(N, dtype=complex)
    pos_freqs = freqs[freqs >= 0]
    H_pos = H[freqs >= 0]
    
    # Interpolate for positive frequencies
    for i, f in enumerate(freqs_full):
        if f >= 0 and f <= np.max(pos_freqs):
            H_full[i] = np.interp(f, pos_freqs, H_pos)
        elif f < 0:
            # Mirror for negative frequencies (conjugate)
            f_pos = abs(f)
            if f_pos <= np.max(pos_freqs):
                H_full[i] = np.conj(np.interp(f_pos, pos_freqs, H_pos))
    
    # Apply frequency response
    vibration_fft = audio_fft * H_full
    
    # Inverse FFT to get predicted vibration signal
    vibration_pred = np.real(ifft(vibration_fft))
    
    return vibration_pred

def process_data(data, fs=16000, nperseg=1024, folder='cache/bcf'):
    """
    Process a single data point to compute frequency response and prediction error.
    
    Parameters:
    data : dict, contains 'audio' and 'vibration' signals
    fs : float, sampling frequency in Hz
    nperseg : int, length of each segment for CSD and PSD
    
    Returns:
    tuple : (H, pred_error), frequency response and mean squared error
    """
    vibration = data['vibration']
    audio = data['audio']
    index = data['index']
    freqs, H = estimate_frequency_response(audio, vibration, fs=fs, nperseg=nperseg)
    pred_vibration = apply_frequency_response(audio, H, freqs, fs=fs)
    pred_error = np.mean((pred_vibration - vibration) ** 2)
    save_dict = {
        'H': H,
        'pred_error': pred_error,
        'freqs': freqs,
        'index': index
    }
    np.savez(f"{folder}/{index}.npz", **save_dict)
    return H, pred_error

class Bone_Conduction_Function():
    def __init__(self, dataset):
        self.dataset = dataset
        self.folder = 'cache'
        self.bcf_folder = f'{self.folder}/bcf'
        self.bcf_files = [f for f in os.listdir(self.bcf_folder) if f.endswith('.npz')]
        print(f"Found {len(self.bcf_files)} cached BCF files in {self.bcf_folder}")

    def extraction(self, num_processes=None):
        '''
        Calculate the frequency response between vibration and audio signals using multiprocessing.
        Iterates through the dataset and extracts features in parallel.
        
        Parameters:
        num_processes : int, number of processes to use (default: number of CPU cores)
        
        Returns:
        Hs : list of tuples, each containing (H, pred_error)
        '''
        if num_processes is None:
            num_processes = mp.cpu_count()  # Use all available CPU cores
        process_func = partial(process_data, fs=16000, nperseg=1024, folder=self.bcf_folder)
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_func, self.dataset), total=len(self.dataset), desc="Processing signals"))

    def aggregation(self):
        """
        Aggregate the frequency responses from all cached files.
        This function assumes that the frequency responses are stored in .npz files.
        """
        pred_errors = []
        for bcf_file in tqdm(self.bcf_files):
            bcf_data = np.load(f'{self.bcf_folder}/{bcf_file}')
            pred_error = bcf_data['pred_error']
            pred_errors.append(pred_error)
        plt.figure(figsize=(10, 5))
        plt.hist(pred_errors, bins=100, color='blue', alpha=0.7)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid()
        plt.savefig(f'{self.folder}/prediction_error_distribution.png')

    def prediction(self, audio):
        bcf_file = np.random.choice(self.bcf_files)
        bcf_data = np.load(f'{self.bcf_folder}/{bcf_file}')
        H = bcf_data['H']; freqs = bcf_data['freqs']
        pred_vibration = apply_frequency_response(audio, H, freqs, fs=16000)
        return pred_vibration



    