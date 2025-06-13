import os
import pandas as pd
import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm
import soundfile as sf
from scipy.signal import find_peaks, correlate

def align4(rx_ori, frame_fmcw, info_band_fmcw, tx, stereo=1, defaultChl=1):
    
    """
    Aligns input signal rx_ori with reference signal tx using cross-correlation and phase alignment.
    
    Parameters:
    -----------
    rx_ori : numpy.ndarray
        Input signal array (2D or 1D)
    frame_fmcw : int
        Frame size for FMCW signal
    info_band_fmcw : array-like
        Indices for frequency band of interest
    tx : numpy.ndarray
        Reference signal for alignment
    stereo : int, optional
        Channel index for output (default: 1)
    defaultChl : int, optional
        Channel index for alignment (default: 1)
    
    Returns:
    --------
    rx : numpy.ndarray
        Aligned signal
    """
    # Compute FFT of reference signal
    txfft = np.fft.fft(tx)
    
    # Select default channel
    rx = rx_ori[:, defaultChl-1] if rx_ori.ndim > 1 else rx_ori
    
    angidelay = []
    
    # Loop for coarse and fine alignment
    for ii in range(min(4, int(np.floor(len(rx) / frame_fmcw) - 1))):
        # Coarse alignment with cross-correlation
        idxtemprx = rx[ii * frame_fmcw:(ii + 2) * frame_fmcw - 1]
        # MATLAB's finddelay is equivalent to finding the lag with max cross-correlation
        corr = correlate(idxtemprx, tx, mode='full')
        lags = np.arange(-len(tx) + 1, len(idxtemprx))
        D = lags[np.argmax(corr)]
        
        if D <= 0:
            D = D + frame_fmcw
        
        # Circular shift for alignment
        jim = np.roll(np.arange(1, frame_fmcw + 1), 25 - D)
        Df = []
        
        # Fine alignment with phase
        for im_t in range(50):
            imfft = np.fft.fft(rx[jim[im_t-1]:jim[im_t-1] + frame_fmcw])
            if imfft.shape[0] != txfft.shape[0]:
                print("warning, rx may be unsperated stereo")
                return np.nan
            
            # Compute phase difference in specified frequency band
            phase_diff = np.sum(np.abs(np.angle(imfft[info_band_fmcw]) - 
                                      np.angle(txfft[info_band_fmcw])))
            Df.append(phase_diff)
        
        # Find minimum phase difference
        angdfft, angidx = np.min(Df), np.argmin(Df)
        angidx = jim[angidx]
        angidelay.append([D, angidx])
    
    # Convert angidelay to array for processing
    angidelay = np.array(angidelay).T
    
    # Find mode of fine alignment indices
    try:
        imidx = int(np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                       axis=1, arr=angidelay[1, :]))
    except Exception as e:
        print(f"Empty angle_based index set {angidelay}")
        imidx = 1  # Fallback to avoid crashing
    
    # Extract aligned signal
    rx = rx_ori[imidx-1:, stereo-1] if rx_ori.ndim > 1 else rx_ori[imidx-1:]
    
    return rx
def synctone(rx):
    """
    sync the 2kHz tone in the received signal.
    """
    Fs = 48000  # sample rate in Hz, adjust as needed
    tone_freq = 2000  # tone frequency in Hz
    tone_duration = 0.5  # duration of tone in seconds, adjust as needed
    detect_duration = 25

    # Create a time vector for one period of the 2kHz tone
    t = np.arange(0, tone_duration, 1/Fs)

    # Create the template signal for the 2kHz tone
    tone = np.sin(2 * np.pi * tone_freq * t)

    # Bandpass filter: 1900-2100 Hz
    sos = butter(4, [1900/(Fs/2), 2100/(Fs/2)], btype='band', output='sos')
    rxlow = sosfiltfilt(sos, rx[:int(Fs * detect_duration)])

    # Take absolute value
    rxlow = np.abs(rxlow)
    tone = np.abs(tone)

    # Cross-correlate the received signal with the tone template
    correlation = np.correlate(rxlow, tone, mode='full')
    lags = np.arange(-len(tone) + 1, len(rxlow))

    # Find the index of maximum correlation
    tone_start_index = np.argmax(np.abs(correlation))

    # Adjust for lags
    tone_start_index = lags[tone_start_index]

    return tone_start_index

def annotation_parser(annotation_file):
    """
    Parse the annotation file to extract relevant information.
    """
    # Read the CSV file
    df = pd.read_csv(annotation_file, header=None)
    # Remove the first row if the first column is 'start'
    start_time = df.iloc[0, 2]
    if df.iloc[0, 0] == 'start':
        df = df.iloc[1:]

    tip_preamable = '##__'
    # remove the row if the first column contains the tip preamble
    df = df[~df.iloc[:, 0].str.contains(tip_preamable, na=False)]

    sentences = []
    for index, row in df.iterrows():
        sentence = row[0]; start = row[1]; end = row[2]
        # Convert start and end times to float
        start = start - start_time; end = end - start_time
        if (end - start) < 0.5:
            end = start + 1
        start = start - 0.1; end = end + 0.1
        sentences.append((sentence, start, end))
    return sentences

def audio_segmentation(audio, sr, sentences, path, frame_length=None):
    """
    Segment the audio signal based on the provided sentences.
    """
    audio_folder = os.path.join(path, 'audio'); os.makedirs(audio_folder, exist_ok=True)
    sentence_txt = []
    for i, (sentence, start, end) in enumerate(tqdm(sentences)):
        # Convert start and end times to sample indices
        start_sample = int(start * sr); end_sample = int(end * sr)
        if frame_length is not None: # Ensure the segment length == multiple of frame_length
            start_sample = (start_sample // frame_length) * frame_length
            end_sample = ((end_sample + frame_length - 1) // frame_length) * frame_length
        segment = audio[:, start_sample:end_sample]
        # Save the segment as a separate file
        segment_filename = os.path.join(audio_folder, f"{i}.wav")
        sf.write(segment_filename, segment.T, sr)
        sentence_txt.append([f"{i}.wav", sentence])
    # Save the sentences to a csv file
    sentence_df = pd.DataFrame(sentence_txt, columns=['filename', 'sentence'])
    sentence_df.to_csv(os.path.join(path, 'sentences.csv'), index=False)
    print(f"Audio segments saved to {audio_folder}")

def EchoBoneDataset(dataset_path = "../dataset/BCH/EchoBone"):
    """
    EchoBone Dataset
    """
    ofdm_frame_file = 'preprocess/ofdm_44100_17822khz_257_paprconfirm_2.wav'
    ofdm_frame, sr = librosa.load(ofdm_frame_file, sr=None, mono=True)

    sessions = os.listdir(dataset_path)
    for session in sessions:
        print(f"Processing session: {session}")
        session_path = os.path.join(dataset_path, session)
        files = os.listdir(session_path)
        csv_files = [f for f in files if f.endswith('.csv')]
        wav_files = [f for f in files if f.endswith('.wav')]
        for csv_file, wav_file in zip(csv_files, wav_files):
            print(f"Processing file: {csv_file}, {wav_file}")
            csv_path = os.path.join(session_path, csv_file)
            wav_path = os.path.join(session_path, wav_file)
            sentences = annotation_parser(csv_path)

            audio, sr = librosa.load(wav_path, sr=None, mono=False)

            # find the start tone in the audio
            audio_tone = audio[:, :sr * 10]
            tone_start_index = synctone(audio_tone[0])
            audio = audio[:, tone_start_index:]

            # # Align the audio with the OFDM frame
            # audio = align4(audio, len(ofdm_frame), np.arange(0, 257), ofdm_frame, stereo=1, defaultChl=1)

            # audio_segmentation(audio, sr, sentences, session_path, frame_length=577)

def BoneVoiceDataset(dataset_path = "../dataset/BCH/BoneVoice"):
    """
    BoneVoice Dataset
    """
    raise NotImplementedError("BoneVoiceDataset function is not implemented yet.")
if __name__ == "__main__":
    # Test the BCHDataset function
    dataset = EchoBoneDataset()
