import gc
import json
import pickle
from collections import OrderedDict

import librosa
import numpy as np
import soundfile as sf
import torch.distributed as dist
from scipy.signal import fftconvolve
from torch.utils.data import Dataset
import torch
from .perturb import (
    WhiteNoisePerturb,
    AACConversionPerturb,
    BandRejectPerturb,
    BassBoostPerturb,
    BitCrushPerturb,
    ColoredNoisePerturb,
    DCOffsetPerturb,
    DRCPerturb,
    EQMuchGainPerturb,
    EQPerturb,
    GSMcodecsPerturb,
    LoudnessPerturb,
    LowPassPerturb,
    MP3CompressorPerturb,
    OPUSCodecsPerturb,
    PacketLossPerturb,
    PitchPerturb,
    SpeakerDistortionPerturbHardClip,
    SpeakerDistortionPerturbHardClipOnRate,
    SpeakerDistortionPerturbPedal,
    SpeakerDistortionPerturbSigmoid1,
    SpeakerDistortionPerturbSigmoid2,
    SpeakerDistortionPerturbSoftClip,
    SpeakerDistortionPerturbSox,
    SpectralLeakagePerturb,
    SpectralTimeFreqHolesPerturb,
    SpeedPerturb,
    LatencyPerturb,
)

class SNR_Controller():
    def __init__(self, snr):
        self.snr = snr

    def __call__(self, audio, noise):
        snr = np.random.uniform(self.snr[0], self.snr[1])
        noise = noise / np.linalg.norm(noise) * np.linalg.norm(audio) / (10 ** (snr / 20))
        noise = noise[:audio.shape[0]]
        audio = audio + noise
        return audio
    
default_perturb_config = {
    'MP3CompressorPerturb': {"vbr_min:": 1, "vbr_max": 9.5},
    'BitCrushPerturb': {"bit_min": 4, "bit_max": 32},
    'PacketLossPerturb': {"loss_rate_min": 0, "loss_rate_max": 0.3, "frame_time_min": 0.008, "frame_time_max": 0.05, 
                          "decay_rate_min": 0, "decay_rate_max": 0.2, "hard_loss_prob": 1.0},
    'WhiteNoisePerturb': {"snr_min": 5, "snr_max": 15},
    'LatencyPerturb': {"min_delay_ms": 20, "max_delay_ms": 100},
}

class Audio_Perturbation():
    def __init__(self, sample_rate, perturb_config=default_perturb_config):
        self.perturb_config = perturb_config
        self.perturb_instances = []
        for class_name, params in perturb_config.items():
            # Get the class from the globals() dictionary
            perturb_class = globals().get(class_name)
            params['sample_rate'] = sample_rate
            if perturb_class is None:
                raise ValueError(f"Class {class_name} not found.")
            # Create an instance of the class with the provided parameters
            perturb_instance = perturb_class(**params)
            # Append the instance to the list
            self.perturb_instances.append([class_name, perturb_instance])
    def __call__(self, audio):
        for perturb_class, perturb_instance in self.perturb_instances:
            audio = perturb_instance(audio) 
            if len(audio.shape) == 1:
                audio = audio[None, :]
        audio = audio.astype(np.float32)
        return audio