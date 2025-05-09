import os
import random
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset
import numpy as np
from .perturb_wrapper import Audio_Perturbation, SNR_Controller

def extract_dvector(dataset_path):
    from resemblyzer import VoiceEncoder, preprocess_wav
    from pathlib import Path
    import numpy as np
    import os
    encoder = VoiceEncoder()
    # dataset_path = '../../dataset/simulation/' + 'earphone_TIMIT_event_2'
    files = os.walk(dataset_path)
    for path, _, file in files:
        for f in file:
            if f.endswith('.wav'):
                fpath = Path(path) / f
                wav = preprocess_wav(fpath)
                embed = encoder.embed_utterance(wav)
                fpath_embed = Path(path) / (f.split('.')[0] + '.npy')
                np.save(fpath_embed, embed)
    
class MixtureDataset(Dataset):
    def __init__(self, speakers_dir, noise_dir, emb_dir, num_speakers=2, sample_rate=16000, duration=5, snr=(-10, 10), relay_config={}, 
                 output_format='separation'):
        """
        Args:
            speakers_dir (str): Path to the directory containing speaker folders.
            noise_dir (list of str): Paths t`o directories containing noise audio files.
            num_speakers (int): Number of speakers to mix in each sample.
            sample_rate (int): Sample rate for audio files.
            duration (int): Duration of the audio clip in seconds.
        """
        self.speakers_dir = speakers_dir
        if noise_dir is None:
            self.noise_files = None
        else:
            # Collect all noise files
            self.noise_files = [os.path.join(noise_dir, file) for file in os.listdir(noise_dir) if file.endswith('.wav') or file.endswith('.flac')]
        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.duration = duration
        self.snr = snr
        self.snr_controller = SNR_Controller(snr)
        self.audio_perturb = Audio_Perturbation(sample_rate, relay_config)
        self.samples_per_file = self.sample_rate * self.duration
        self.task = output_format

        # Collect all speaker folders
        self.mixture_files = self._load_mixture_files(speakers_dir)
        self.embeddings = self._load_embeddings(emb_dir)

    def _load_embeddings(self, emb_dir):
        speakers = os.listdir(emb_dir)
        embeddings = {}
        for i, speaker in enumerate(speakers):
            speaker_embeddings = torch.load(os.path.join(emb_dir, speaker), weights_only=False)
            file_names = speaker_embeddings.keys()
            for file_name in file_names:
                speaker_embedding = speaker_embeddings[file_name]
                embeddings[file_name] = speaker_embedding
        return embeddings
        
    def _load_mixture_files(self, speakers_dir, n=None, seed=42):
        speaker_folders = [os.path.join(speakers_dir, folder) for folder in os.listdir(speakers_dir) if os.path.isdir(os.path.join(speakers_dir, folder))]
        speaker_files = {}
        num_audio_files = 0
        for folder in speaker_folders:
            files = []
            for root, dirs, _files in os.walk(folder):
                for file in _files:
                    if file.endswith('.flac'):
                        files.append(os.path.join(root, file))
            speaker_files[folder] = files
            num_audio_files += len(files)
        if n is None:
            n = num_audio_files
        if seed is not None:
            random.seed(seed)  # Set the random seed for reproducibility 
        mixture_files = [] 
        for _ in range(n):
            selected_speakers = random.sample(speaker_folders, self.num_speakers)
            mixture_file = []
            for speaker_folder in selected_speakers:
                # pick one speaker file and one enrollment file
                mixture_file.append(random.sample(speaker_files[speaker_folder], 2))
            mixture_files.append(mixture_file)

        return mixture_files

    def __len__(self):
        return len(self.mixture_files)  # You can adjust this based on your needs

    def __random_noise__(self):
        if self.noise_files is None:
            noise_waveform = np.zeros((self.samples_per_file))
        else:
            noise_file = random.choice(self.noise_files)
            noise_waveform, _ = librosa.load(noise_file, sr=self.sample_rate)
            # Ensure the waveform is of the correct length
            if len(noise_waveform) < self.samples_per_file:
                noise_waveform = np.pad(noise_waveform, (0, self.samples_per_file - len(noise_waveform)))
            # Trim to the desired duration
            noise_waveform = noise_waveform[:self.samples_per_file]
        return noise_waveform
    
    def __getitem__(self, idx):
        mixture_file = self.mixture_files[idx]
        # Sample speakers
        sources = []; embeddings = []
        for (speaker_file, enrollment_file) in mixture_file:
            waveform, _  = librosa.load(speaker_file, sr=self.sample_rate)
            # Ensure the waveform is of the correct length
            if len(waveform) < self.samples_per_file:
                waveform = np.pad(waveform, (0, self.samples_per_file - len(waveform)))
            # Trim to the desired duration
            sources.append(waveform[:self.samples_per_file])

            # embedding = self.embeddings[os.path.basename(speaker_file)] # Use the speaker file name as the key
            embedding = self.embeddings[os.path.basename(enrollment_file)] # Use the enrollment file name as the key
            embeddings.append(embedding)

        sources = np.stack(sources); embeddings = np.stack(embeddings)

        # noise_waveform = self.__random_noise__() # external noise

        if self.task == 'separation': # permutation-possible separation
            mixture = sum(sources)
            mixture = mixture / np.max(np.abs(mixture))
            return {'mixture': mixture, 'sources': sources}
        else:
            random_idx = np.random.randint(0, self.num_speakers - 1)
            embeddings = embeddings[random_idx:random_idx+1]
            if self.task in ['extraction', 'relay']:
                keep_idx = [i for i in range(self.num_speakers) if i == random_idx]
            else:
                keep_idx = [i for i in range(self.num_speakers) if i != random_idx]
            mixture = sum(sources); sources = sources[keep_idx]
            if len(keep_idx) > 1: # If more than one speaker is selected, sum them up
                sources = np.sum(sources, axis=0, keepdims=True)
            interference = mixture - sources
            mixture = self.snr_controller(sources, interference)
            if self.task == 'relay':
                relay = self.audio_perturb(sources)
                return {'mixture': mixture, 'sources': sources, 'embeddings': embeddings, 'relay': relay}
            else:
                return {'mixture': mixture, 'sources': sources, 'embeddings': embeddings}


if __name__ == '__main__':
    speakers_dir = 'dataset/MixLibriSpeech/LibriSpeech/dev-clean'
    noise_dir = 'dataset/MixLibriSpeech/wham_noise/tt'
    emb_dir = 'dataset/MixLibriSpeech/librispeech_dvector_embeddings/dev-clean'
    dataset = MixtureDataset(speakers_dir, noise_dir, emb_dir, num_speakers=3, output_format='extraction', sample_rate=16000, duration=5)
    print(len(dataset))  # Number of samples in the dataset
    data_sample = dataset[0]  # Get a sample from the dataset
    mixture = data_sample['mixture']; sources = data_sample['sources']; embeddings = data_sample['embeddings']
    print(mixture.size(), sources.size(), embeddings.size())  # Print the shapes of the sample tensors