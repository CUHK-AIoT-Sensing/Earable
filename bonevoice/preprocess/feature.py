import numpy as np
import librosa
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
import scipy

import scipy.signal as signal
import seaborn as sns

def extract_reflected_features(rx, letter, frame = 2046, sr = 48000, hop_length_modifier = 22, plot_vmax = .05):
    spec = librosa.stft(rx, n_fft = frame, hop_length = frame//hop_length_modifier ,window=signal.windows.hann(frame), center=False)
    spec = np.abs(spec)
    plt_total = np.zeros((spec.shape))

    min_len = np.inf
    for i in range(hop_length_modifier):
        test = spec[:,i::hop_length_modifier]
        test_1 = np.diff(test,1,axis=1)
        r1 = test_1.shape[1]
        r2 = plt_total[:,i::hop_length_modifier].shape[1]
        if r1 < min_len:
            min_len = r1
        if r2 < min_len:
            min_len = r2

    for i in range(hop_length_modifier):
        test = spec[:,i::hop_length_modifier]
        test_1 = np.diff(test,1,axis=1)
        plt_total[:,i::hop_length_modifier][:,:min_len] = test_1[:,:min_len]
    return plt_total


def slwinAR(rx, frame, ar_lags, var1, speechlen, stride, ifvar):
    """
    Auxiliary function for sliding window AR coefficient estimation.
    
    Parameters:
    -----------
    rx : numpy.ndarray 
        Input signal array
    frame_full : float
        Frame size (can be non-integer)
    ar_lags : int
        Number of lags for AR model
    var1 : int
        Order of differencing
    speechlen : int
        Length of speech segment in frames
    stride : float
        Stride for sliding window
    ifvar : bool
        If True, use Vector AR (VAR); else use univariate AR
    
    Returns:
    --------
    fea : list
        List containing AR coefficients
    """
    # Calculate window parameters
    chan_wid = int(np.floor(len(rx) / max(frame, frame)))
    Ttmp = int(np.floor((chan_wid - speechlen) / stride) + 1)    
    
    # Check Ttmp
    if Ttmp <= 0:
        print("Ttmp <= 0")
        return [[]]
    
    # Initialize output array
    rxset = np.zeros((Ttmp, speechlen * frame, rx.shape[1]))
    
    # Compute differences
    d_rx_all = np.diff(np.vstack([rx, rx[:var1, :]]), n=var1, axis=0)
    
    # Sliding window
    for ii in range(Ttmp):
        idx_start = round(ii * stride * frame)
        idx_end = idx_start + speechlen * frame
        d_rx = d_rx_all[idx_start:idx_end, :]
        try:
            if d_rx.ndim > 1:
                rxset[ii, :, :] = d_rx
            else:
                rxset[ii, :] = d_rx
        except Exception as e:
            print("cat error")
    
    # Autoregressive modeling
    if not ifvar:
        # Univariate AR for each channel
        arma_coef = np.full((ar_lags, 2), np.nan)
        for ii in range(Ttmp):
            arma_down = AutoReg(rxset[ii, :, 0], lags=ar_lags).fit().params[1:]  # Skip intercept
            arma_up = AutoReg(rxset[ii, :, 1], lags=ar_lags).fit().params[1:]
            arma_coef[:, 0] = arma_down
            arma_coef[:, 1] = arma_up
    else:
        # Vector AR (VAR) modeling
        arma_coef = np.full((Ttmp, ar_lags, 4), np.nan)
        for ii in range(Ttmp):
            # Extract the current window
            _var = VAR(rxset[ii, :, :2])  # Assuming 2 channels
            _var = _var.fit(maxlags=ar_lags, trend='n')  # No trend
            # Extract AR coefficients (reshape to match MATLAB's output)
            ar_params = _var.params.T  # Shape: (lags * nvars^2, )
            ar_params = ar_params[:ar_lags * 4].reshape(ar_lags, 4)
            arma_coef[ii, :, :] = ar_params

    return arma_coef

class FeatureExtractor():
    def __init__(self, dataset_path = "../dataset/Bone", fs=48000, n_fft=2048, hop_length=512, win_length=2048):
        self.dataset_path = dataset_path
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def extract_dataset(self, feature_type="ar"):
        """
        Extract features from the dataset.
        """
        sessions = os.listdir(self.dataset_path)
        for session in sessions:
            session_path = os.path.join(dataset_path, session)
            
            features_folder = os.path.join(session_path, "features")
            os.makedirs(features_folder, exist_ok=True)

            audio_path = os.path.join(session_path, "audio")
            sentences_path = os.path.join(session_path, "sentences.csv")
            sentences = pd.read_csv(sentences_path, header=[0])
            for index, row in sentences.iterrows():
                audio_file, sentence = row['filename'], row['sentence']
                audio, fs = librosa.load(os.path.join(audio_path, audio_file), sr=self.fs, mono=False)
                # from (Channel, Time) to (Time, Channel)
                audio = np.transpose(audio)
                features = self.extract_features(audio, feature_type)
                # Save features to file
                feature_file = os.path.join(features_folder, f"{index}.npy")
                np.save(feature_file, features)
    
    def extract_features(self, audio, feature_type):
        if feature_type == "ar":
            # Example feature extraction using AR model
            ar_lags = 100
            frame_full = 577
            var1 = 2
            speechlen = 6
            stride = 3
            ifvar = True
            # high-pass filter above 17000Hz
            b, a = scipy.signal.butter(4, 17000, 'hp', fs=self.fs)
            audio = scipy.signal.filtfilt(b, a, audio, axis=0)
            features = slwinAR(audio, frame_full, ar_lags, var1, speechlen, stride, ifvar)
            print("Extracted features:", features.shape)
        else:
            print("Feature type not supported.")
        return features
    
if __name__ == "__main__":
    dataset_path = "../dataset/BCH/EchoBone"
    feature_extractor = FeatureExtractor(dataset_path)
    feature_extractor.extract_dataset()