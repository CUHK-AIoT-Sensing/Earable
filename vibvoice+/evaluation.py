import scipy.signal as signal
import numpy as np
import torch
from pesq import pesq, pesq_batch
from joblib import Parallel, delayed
from pystoi.stoi import stoi
def eval(clean, predict):
    metric1 = batch_pesq(clean, predict, 'wb')
    metric2 = batch_pesq(clean, predict, 'nb')
    metric3 = SI_SDR(clean, predict)
    metric4 = batch_stoi(clean, predict)
    metrics = [metric1, metric2, metric3, metric4]
    return np.stack(metrics, axis=1)

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

def LSD(gt, est):
    spectrogram1 = np.abs(signal.stft(gt, fs=16000, nperseg=640, noverlap=320, axis=1)[-1])
    spectrogram2 = np.abs(signal.stft(est, fs=16000, nperseg=640, noverlap=320, axis=1)[-1])
    error = np.log10(spectrogram1) - np.log10(spectrogram2)
    error = np.mean(error ** 2, axis=(1, 2)) ** 0.5
    return error

def batch_pesq(clean, noisy, mode):
    pesq_score = Parallel(n_jobs=-1)(delayed(pesq)(16000, c, n, mode, on_error=1) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    return pesq_score

def batch_stoi(clean, noisy):
    stoi_score = Parallel(n_jobs=-1)(delayed(stoi)(c, n, 16000) for c, n in zip(clean, noisy))
    stoi_score = np.array(stoi_score)
    return stoi_score

def batch_ASR(batch, asr_model):
    batch_size = batch.shape[0]
    wav_lens = torch.ones(batch_size)
    pred = asr_model.transcribe_batch(batch, wav_lens)[0]
    return pred
def eval_ASR(clean, noisy, text, asr_model):
    clean = torch.from_numpy(clean/np.max(clean, axis=1)[:, np.newaxis]) * 0.8
    noisy = torch.from_numpy(noisy/np.max(noisy, axis=1)[:, np.newaxis]) * 0.8
    pred_clean = batch_ASR(clean, asr_model)
    pred_noisy = batch_ASR(noisy, asr_model)
    wer_clean = []
    wer_noisy = []
    for p_c, p_n, t in zip(pred_clean, pred_noisy, text):
        wer_clean.append(wer(t.split(), p_c.split()))
        wer_noisy.append(wer(t.split(), p_n.split()))
    return wer_clean, wer_noisy
if __name__ == "__main__":
    # we evaluate WER and PESQ in this script
    f = open('survey/survey.txt', 'r', encoding='UTF-8')
    lines = f.readlines()
    WER = []
    for i in range(len(lines)):
        hy = lines[i].upper().split()
        if len(hy) < 3:
            continue
        gt = sentences[i // int(len(lines) / len(sentences))]
        WER.append(wer(gt, hy))
    print(np.mean(WER))

    # from speechbrain.pretrained import EncoderDecoderASR
    # # Uncomment for using another pre-trained model
    # asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
    #                                            savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
    #                                            run_opts={"device": "cuda"})
    # text = [["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"],
    #         ["HAPPY", "NEW", "YEAR", "PROFESSOR", "AUSTIN", "NICE", "TO", "MEET", "YOU"]]
    # clean = torch.zeros([2, 80000])
    # noisy = torch.zeros([2, 80000])
    # eval_ASR(clean, noisy, text, asr_model)


