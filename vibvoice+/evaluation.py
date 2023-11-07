import numpy as np
import torch
from pesq import pesq, pesq_batch
from joblib import Parallel, delayed
from pystoi.stoi import stoi
import librosa
from skimage.metrics import structural_similarity as ssim

EPS = 1e-12
def eval(clean, predict):
    metrics = []
    metrics.append(batch_pesq(clean, predict, 'wb'))
    metrics.append(batch_pesq(clean, predict, 'nb'))
    metrics.append(SI_SDR(clean, predict))
    metrics.append(batch_stoi(clean, predict))
    return np.stack(metrics, axis=1)

def to_log(input):
    return np.log10(input + 1e-12)
class AudioMetrics:
    def __init__(self, rate):
        self.rate = rate
        self.hop_length = int(rate / 100)
        self.n_fft = int(2048 / (44100 / rate))

    def read(self, est, target):
        est, _ = librosa.load(est, sr=self.rate, mono=True)
        target, _ = librosa.load(target, sr=self.rate, mono=True)
        return est, target

    def wav_to_spectrogram(self, wav):
        f = np.abs(librosa.stft(wav, hop_length=self.hop_length, n_fft=self.n_fft))
        f = np.transpose(f, (1, 0))
        return f

    def center_crop(self, x, y):
        dim = 2
        if x.size(dim) == y.size(dim):
            return x, y
        elif x.size(dim) > y.size(dim):
            offset = x.size(dim) - y.size(dim)
            start = offset // 2
            end = offset - start
            x = x[:, :, start:-end, :]
        elif x.size(dim) < y.size(dim):
            offset = y.size(dim) - x.size(dim)
            start = offset // 2
            end = offset - start
            y = y[:, :, start:-end, :]
        assert (
            offset < 10
        ), "Error: the offset %s is too large, check the code please" % (offset)
        return x, y
    def batch_evaluation(self, batch_est, batch_target):
        result = []
        for est, target in zip(batch_est, batch_target):
            result.append(self.evaluation(est, target))
        return result
    def evaluation(self, est, target):
        """evaluate between two audio
        Args:
            est (str or np.array): _description_
            target (str or np.array): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # import time; start = time.time()
        if type(est) != type(target):
            raise ValueError(
                "The input value should either both be numpy array or strings"
            )
        if type(est) == type(""):
            est_wav, target_wav = self.read(est, target)
        else:
            assert len(list(est.shape)) == 1 and len(list(target.shape)) == 1, (
                "The input numpy array shape should be [samples,]. Got input shape %s and %s. "
                % (est.shape, target.shape)
            )
            est_wav, target_wav = est, target

        # target_spec_path = os.path.join(os.path.dirname(file), os.path.splitext(os.path.basename(file))[0]+"_proc_%s.pt" % (self.rate))
        # if(os.path.exists(target_spec_path)):
        #     target_sp = torch.load(target_spec_path)
        # else:

        assert (
            abs(target_wav.shape[0] - est_wav.shape[0]) < 100
        ), "Error: Shape mismatch between target and estimation %s and %s" % (
            str(target_wav.shape),
            str(est_wav.shape),
        )

        min_len = min(target_wav.shape[0], est_wav.shape[0])
        target_wav, est_wav = target_wav[:min_len], est_wav[:min_len]

        target_sp = self.wav_to_spectrogram(target_wav)
        est_sp = self.wav_to_spectrogram(est_wav)
        # frequency domain, lsd, log_sispec, sispec, ssim
        result = [self.lsd(est_sp, target_sp),
        self.sispec(
            to_log(est_sp), to_log(target_sp)
        ),
        self.sispec(est_sp, target_sp),
        self.ssim(est_sp, target_sp),]

        return result

    def lsd(self, est, target):
        lsd = np.log10(target**2 / ((est + EPS) ** 2) + EPS) ** 2
        lsd = np.mean(np.mean(lsd, axis=-1) ** 0.5, axis=-1)
        return lsd

    def sispec(self, est, target):
        # in log scale
        output, target = energy_unify(est, target)
        noise = output - target
        sp_loss = 10 * np.log10(
            (pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
        )
        return np.sum(sp_loss) / sp_loss.shape[0]

    def ssim(self, est, target):
        res = ssim(target,est, win_size=7,data_range=est.max() - est.min())
        return res


def pow_p_norm(signal):
    """Compute 2 Norm"""
    signal = signal.reshape(-1)
    return np.linalg.norm(signal, ord=2, keepdims=True)**2


def energy_unify(estimated, original):
    target = pow_norm(estimated, original) * original
    target /= pow_p_norm(original) + EPS
    return estimated, target


def pow_norm(s1, s2):
    return np.sum(s1 * s2, keepdims=True)

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

def LSD(target, est):
    lsd = torch.log10(target**2 / ((est + 1e-8) ** 2) + 1e-8) ** 2
    lsd = torch.mean(torch.mean(lsd, dim=1) ** 0.5, dim=1)
    return lsd

def MAE(clean, predict):
    error = np.abs(clean - predict).mean(axis=(1, 2))
    return error

def batch_pesq(clean, noisy, mode):
    pesq_score = Parallel(n_jobs=-1)(delayed(pesq)(16000, c, n, mode, on_error=1) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    return pesq_score

def batch_stoi(clean, noisy):
    stoi_score = Parallel(n_jobs=-1)(delayed(stoi)(c, n, 16000) for c, n in zip(clean, noisy))
    stoi_score = np.array(stoi_score)
    return stoi_score


