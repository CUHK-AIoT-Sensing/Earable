import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc
rc('text', usetex=False)
plt.rcParams.update({'font.size': 20})
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
from .vib_dataset import EMSB_dataset, ABCS_dataset, V2S_dataset


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
    if folder is None:
        return H, pred_error, pred_vibration
    save_dict = {
        'H': H,
        'pred_error': pred_error,
        'freqs': freqs,
        'index': index
    }
    np.savez(f"{folder}/{index}.npz", **save_dict)
    return H, pred_error, pred_vibration

def dataset_parser(dataset_name, split='all'):
    if dataset_name == 'ABCS':
        dataset = ABCS_dataset(split=split)
    elif dataset_name == 'EMSB':
        dataset = EMSB_dataset(split=split)
    elif dataset_name == 'V2S':
        dataset = V2S_dataset(split=split)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset

class Bone_Conduction_Function():
    def __init__(self, dataset_name='ABCS', folder = 'data'):
        self.dataset_name = dataset_name
        self.dataset = dataset_parser(dataset_name)
        self.folder = folder
        self.bcf_folder = f'{self.folder}/{dataset_name}'
        if not os.path.exists(self.bcf_folder):
            os.makedirs(self.bcf_folder)
            print(f"Created folder: {self.bcf_folder}")
        self.load_bcf()  # Load existing BCF data if available

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

        splits = self.dataset.json_file.keys()
        for i, split in enumerate(splits):
            self.dataset.split_dataset(split)
            print(f"Processing split {i+1}/{len(splits)}: {split}, {len(self.dataset)} examples")
            bcf_folder = f'{self.bcf_folder}/{i}'
            if not os.path.exists(bcf_folder):
                os.makedirs(bcf_folder)
            # only use randomly select 100 examples for each split
            if len(self.dataset) > 100:
                self.dataset.dataset = np.random.choice(self.dataset.dataset, size=100, replace=False).tolist()
            else:
                print(f"Warning: Split {split} has less than 100 examples, using all {len(self.dataset)} examples.")
            process_func = partial(process_data, fs=16000, nperseg=1024, folder=bcf_folder)
            with mp.Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap(process_func, self.dataset), total=len(self.dataset), desc="Processing signals"))
    
    def load_bcf(self):
        splits = os.listdir(self.bcf_folder)
        self.BCF = {}
        for i, split in enumerate(splits):
            PSDs = []
            split_folder = f'{self.bcf_folder}/{split}'
            bcf_files = [f for f in os.listdir(split_folder) if f.endswith('.npz')]
            for bcf_file in bcf_files:
                bcf_data = np.load(f'{split_folder}/{bcf_file}')
                PSD = bcf_data['H']
                PSDs.append(PSD)
            self.BCF[split] = PSDs
        self.freqs = bcf_data['freqs']  # Assuming all splits have the same freqs
        print(f"Loaded BCF data for {len(splits)} splits from {self.bcf_folder}")

    def predict(self, audio):
        splits = self.BCF.keys()
        split = np.random.choice(list(splits))  # Randomly select a split
        bcfs = self.BCF[split]; bcfs = np.array(bcfs)
        mean_bcf = np.mean(bcfs, axis=0)
        # random_index = np.random.randint(0, len(bcfs))
        # random_bcf = bcfs[random_index]
        pred_vibration = apply_frequency_response(audio, mean_bcf, self.freqs, fs=16000)
        return mean_bcf, pred_vibration

    def plot_reconstruction_error(self):
        splits = self.dataset.json_file.keys()
        for i, split in enumerate(splits):
            self.dataset.split_dataset(split)
            bcf_folder = f'{self.bcf_folder}/{i}'
            bcf_files = [f for f in os.listdir(bcf_folder) if f.endswith('.npz')]
            print(f"Processing split {i+1}/{len(splits)}: {split}, {len(bcf_files)} examples")
            pred_errors = []
            for bcf_file in bcf_files:
                bcf_data = np.load(f'{bcf_folder}/{bcf_file}')
                pred_error = bcf_data['pred_error']
                pred_errors.append(pred_error)
            mean_pred_error = np.mean(pred_errors); std_pred_error = np.std(pred_errors)
            plt.bar([i], [mean_pred_error], yerr=std_pred_error/(len(pred_errors)**0.5), color='blue', alpha=0.5, label='Mean Prediction Error')
        plt.savefig(f'{self.folder}/pred_error_{self.dataset_name}.png')

    def plot_reconstruction(self, index, fs=16000, nperseg=1024):
        data = self.dataset[index]
        vibration = data['vibration']
        audio = data['audio']
        H, pred_error, pred_vibration = process_data(data, fs=fs, nperseg=nperseg, folder=None)

        fig, axs = plt.subplots(3, 2, figsize=(10, 6))
        axs[0, 0].plot(vibration, label='Vibration Signal', color='blue')
        axs[0, 0].set_title('Vibration Signal')
        axs[0, 1].specgram(vibration, Fs=fs, NFFT=nperseg, noverlap=nperseg//2, cmap='viridis')
        axs[0, 1].set_title('Vibration Spectrogram')
        axs[1, 0].plot(pred_vibration, label='Predicted Vibration', color='green')
        axs[1, 0].set_title('Predicted Vibration Signal')
        axs[1, 1].specgram(pred_vibration, Fs=fs, NFFT=nperseg, noverlap=nperseg//2, cmap='viridis')
        axs[1, 1].set_title('Predicted Vibration Spectrogram')
        axs[2, 0].plot(audio, label='Audio Signal', color='red')
        axs[2, 0].set_title('Audio Signal')
        axs[2, 1].specgram(audio, Fs=fs, NFFT=nperseg, noverlap=nperseg//2, cmap='viridis')
        axs[2, 1].set_title('Audio Spectrogram')
        plt.tight_layout()
        fig.savefig(f'{self.folder}/reconstruction_{index}_{self.dataset_name}.png')

    def plot_bcf(self):
        splits = os.listdir(self.bcf_folder)
        splits = splits[:5]
        # sample the same number of colors for each split
        color_list = plt.cm.viridis(np.linspace(0, 1, len(splits)))
        plt.figure(figsize=(10, 5))
        for i, split in enumerate(splits):
            PSDs = self.BCF[split]
            mean_PSD = np.mean(np.abs(PSDs), axis=0); std_PSD = np.std(np.abs(PSDs), axis=0)
            plt.plot(mean_PSD, label='Mean Frequency Response', color=color_list[i])
            plt.fill_between(range(len(mean_PSD)), mean_PSD - std_PSD, mean_PSD + std_PSD, color=color_list[i], alpha=0.2, label='Standard Deviation')
        freqs = self.freqs
        plt.xticks(ticks=np.arange(0, len(freqs), step=100), labels=np.round(freqs[::100], 2), rotation=45)
        plt.title('Bone Conduction Function')
        plt.xlabel('Frequency')
        plt.xscale('log')
        plt.ylabel('Magnitude')
        plt.tight_layout()
        plt.savefig(f'resources/bcf_{self.dataset_name}.pdf')

