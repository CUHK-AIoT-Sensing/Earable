import torch
import numpy as np
from evaluation import batch_pesq, SI_SDR, batch_stoi
from feature import stft, istft
from skimage import filters
import time
def eval(clean, predict):
    metric1 = batch_pesq(clean, predict, 'wb')
    metric2 = batch_pesq(clean, predict, 'nb')
    metric3 = SI_SDR(clean, predict)
    metric4 = batch_stoi(clean, predict)
    metrics = [metric1, metric2, metric3, metric4]
    return np.stack(metrics, axis=1)

def MixIT_loss(yhat, mix_stft, noise_stft, noisy_stft):
    
    """
    The loss for mixture invariant training (MixIT). 
    
    Args:
        yhat: The output of the convolutional neural network before the non-linear activation function. There are three output
            channels corresponding to an enhanced speech and two noise estimates.
        mix_stft: The mixture of a noisy signal example and a noise example in the short-time Fourier transform domain.
        noise_stft: The noise example in the short-time Fourier transform domain.
        noisy_stft: The noisy speech example in the short-time Fourier transform domain.
        
    Returns:
        The loss.
    """

    # loss1 = Spectral_Loss(torch.sigmoid(yhat[:,0,:,:]) + torch.sigmoid(yhat[:,1,:,:]) * mix_stft, noisy_stft)
    # loss1 += Spectral_Loss(torch.sigmoid(yhat[:,2,:,:])  * mix_stft, noise_stft)
    # loss2 = Spectral_Loss(torch.sigmoid(yhat[:,0,:,:]) + torch.sigmoid(yhat[:,2,:,:]) * mix_stft, noisy_stft)
    # loss2 += Spectral_Loss(torch.sigmoid(yhat[:,1,:,:])  * mix_stft, noise_stft)

    loss1 = torch.mean(((torch.sigmoid(yhat[:,0,:,:]) + torch.sigmoid(yhat[:,1,:,:])) * torch.abs(mix_stft) - torch.abs(noisy_stft)) ** 2)
    loss1 += torch.mean((torch.sigmoid(yhat[:,2,:,:])  * torch.abs(mix_stft) - torch.abs(noise_stft)) ** 2)
    loss2 = torch.mean(((torch.sigmoid(yhat[:,0,:,:]) + torch.sigmoid(yhat[:,2,:,:])) * torch.abs(mix_stft) - torch.abs(noisy_stft)) ** 2)
    loss2 += torch.mean((torch.sigmoid(yhat[:,1,:,:])  * torch.abs(mix_stft) - torch.abs(noise_stft)) ** 2)
    
    return torch.minimum(loss1, loss2)

def sigmoid_loss(z):
    
    """
    The sigmoid loss for binary classification based on empirical risk minimization. See R. Kiryo, G. Niu, 
    M. C. du Plessis, and M. Sugiyama, “Positive-unlabeled learning with non-negative risk estimator,” in 
    Proc. NIPS, CA, USA, Dec. 2017.
    
    Args:
        z: The margin.
    
    Returns:
        The loss value.
    """

    return torch.sigmoid(-z)

def weighted_pu_loss(y, yhat, mix_stft, beta, gamma = 1.0, prior = 0.5, ell = sigmoid_loss, p = 1, mode = 'nn'):
    
    """
    The weighted loss for learning from positive and unlabelled data (PU learning). See N. Ito and M. Sugiyama, 
    "Audio signal enhancement with learning from positive and unlabelled data," arXiv, 
    https://arxiv.org/abs/2210.15143. 
    
    Args:
        y: A mask indicating whether each time-frequency component is positive (1) or unlabelled (0).
        yhat: The output of the convolutional neural network before the non-linear activation function.
        mix_stft: The noisy speech in the short-time Fourier transform domain.
        beta: The beta parameter in PU learning using non-negative empirical risk.
        gamma: The gamma parameter in PU learning using non-negative empirical risk.
        prior: The class prior for the positive class.
        ell: The loss function for each time-frequency component such as the sigmoid loss.
        p: The exponent for the weight. p = 1.0 corresponds to weighting by the magnitude spectrogram of the 
            input noisy speech. p = 0.0 corresponds to no weighting.
        mode: The type of the empirical risk in PU learning. 'nn' corresponds to the non-negative empirical 
            risk. 'unbiased' corresponds to the unbiased empirical risk.
        
    Returns:
        The weighted loss.
    """
    
    # Note for future updates: Divide this into two functions, like weighted_sigmoid_loss and TF_level_loss?
    
    epsilon = 10 ** -7
    weight = (torch.abs(mix_stft) + epsilon) ** p
    pos = prior * torch.sum(y * ell(yhat) * weight) / (torch.sum(y) + epsilon)
    neg = torch.sum((1. - y) * ell(-yhat) * weight) / (torch.sum(1. - y) + epsilon) - prior * torch.sum(y * ell(-yhat) * weight) / (torch.sum(y) + epsilon)

    if mode == 'unbiased':
        loss = pos + neg
    elif mode == 'nn':
        if neg > beta:
            loss = pos + neg
        else:
            loss = gamma * (beta - neg)
    return loss
def get_mask(acc, vad):
    '''
    1. 
    noise -> inactivity -> mask = 1 (determined by ~vad)
    the others (unlabelled) -> mask = 2
    2. 
    acc -> activity -> mask of 1
    others (include nothing, noise, speech) -> inactivity -> all 0
    '''
    mask = torch.zeros_like(acc)
    mask = torch.masked_fill(mask, ~vad.bool(), 1)
    ratio = 1 - torch.mean(vad)
    # threshold = filters.threshold_otsu(acc.numpy())
    # mask = (acc > threshold).to(dtype=torch.float32)
    return mask, ratio
def Spectral_Loss(x_mag, y_mag, vad=1):
    """Calculate forward propagation.
          Args:
              x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
              y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
              vad (Tensor): VAD of groundtruth signal (B, #frames, #freq_bins).
          Returns:
              Tensor: Spectral convergence loss value.
          """
    x_mag = torch.clamp(x_mag, min=1e-7)
    y_mag = torch.clamp(y_mag, min=1e-7)
    spectral_convergenge_loss =  torch.norm(vad * (y_mag - x_mag), p="fro") / torch.norm(y_mag, p="fro")
    log_stft_magnitude = (vad * (torch.log(y_mag) - torch.log(x_mag))).abs().mean()
    return 0.5 * spectral_convergenge_loss + 0.5 * log_stft_magnitude

def train_DPCRN(model, sample, optimizer, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, _, _, _ = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    optimizer.zero_grad()
    acc, _, _, _ = stft(acc, 640, 320, 640)
    predict_mask = model(noisy_mag.to(device), acc.to(device))
    loss = Spectral_Loss(predict_mask * noisy_mag.to(device), clean_mag.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()
def test_DPCRN(model, sample, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    predict_mask = model(noisy_mag.to(device), acc.to(device))
    predict = (predict_mask.cpu() * noisy_mag).cpu()
    predict = istft((predict.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy().squeeze(1)
    return clean, predict

def train_SUB_DPCRN(model, sample, optimizer, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, _, _, _ = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    optimizer.zero_grad()
    acc, _, _, _ = stft(acc, 640, 320, 640)
    acc = torch.norm(acc, dim=1, p=2, keepdim=True)
    predict_mask = model(noisy_mag.to(device), acc.to(device))
    loss = Spectral_Loss(predict_mask * noisy_mag.to(device), clean_mag.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()
def test_SUB_DPCRN(model, sample, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    acc = torch.norm(acc, dim=1, p=2, keepdim=True)
    predict_mask = model(noisy_mag.to(device), acc.to(device))
    predict = (predict_mask.cpu() * noisy_mag)
    predict = istft((predict.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy().squeeze(1)
    return clean, predict

def train_PULSE(model, sample, optimizer, device='cuda'):
    optimizer.zero_grad()
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, _, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    # acc = torch.mean(acc, dim=1, keepdim=True)
    vad = vad.reshape(vad.shape[0], 1, 1, -1)

    # mask, ratio = get_mask(acc, vad)
    mask = sample['mask'].reshape(-1, 1, 1, 1) * torch.ones(noisy_mag.size(), dtype = torch.float32); ratio = 0.6

    predict_mask = model(noisy_mag.to(device), acc.to(device))
    loss = weighted_pu_loss(mask.to(device), predict_mask, noisy_mag.to(device), beta=0, gamma=0.096, prior=ratio, p=1.0, mode='nn')
    loss.backward()
    optimizer.step()
    return loss.item()

def test_PULSE(model, sample, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    acc = torch.norm(acc, dim=1, p=2, keepdim=True)
    predict_mask = model(noisy_mag.to(device), acc.to(device))
    predict = (torch.sigmoid(-predict_mask).cpu() * noisy_mag)
    predict = istft((predict.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy().squeeze(1)
    return clean, predict

def train_MIXIT(model, sample, optimizer, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; 
    vad = sample['vad']; noise = sample['noise']; mixture = sample['mixture']

    mixture_mag, _, _, _ = stft(mixture, 640, 320, 640)
    noisy_mag, _, _, _ = stft(noisy, 640, 320, 640)
    noise_mag, _, _, _ = stft(noise, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    vad = vad.reshape(vad.shape[0], 1, 1, -1)

    optimizer.zero_grad()
    predict_mask = model(mixture_mag.to(device) ** 1/15, acc.to(device))
    loss = MixIT_loss(predict_mask, mixture_mag.to(device), noise_mag.to(device), noisy_mag.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def test_MIXIT(model, sample, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']; 

    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    vad = vad.reshape(vad.shape[0], 1, 1, -1)

    predict_mask = model(noisy_mag.to(device) ** 1/15, acc.to(device))
    predict = noisy_mag * torch.sigmoid(predict_mask[:, :1, :, :]).cpu()
    predict = istft((predict.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy().squeeze(1)
    return clean, predict

def train_Skip_DPCRN(model, sample, optimizer, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, _, _, _ = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    acc = torch.norm(acc.to(device=device), dim=1, p=2, keepdim=True)
    vad = vad.to(device=device).reshape(vad.shape[0], 1, 1, -1)

    predict_mask, predict_vad = model.vad_forward_train(noisy_mag, acc)
    # loss_vad = torch.nn.functional.binary_cross_entropy(predict_vad.squeeze(), vad.to(device=device))
    loss_se = Spectral_Loss(predict_mask * noisy_mag, clean_mag)
    loss = loss_se * 1
    loss.backward()
    optimizer.step()
    return loss.item()

def test_Skip_DPCRN(model, sample, device='cuda'):
    acc = sample['imu']; noisy = sample['noisy']; clean = sample['clean']; vad = sample['vad']
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    noisy_mag = noisy_mag.to(device=device)
    clean_mag = clean_mag.to(device=device)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    acc = torch.norm(acc.to(device=device), dim=1, p=2, keepdim=True)
    vad = vad.to(device=device).reshape(vad.shape[0], 1, 1, -1)
    
    predict_mask = model.vad_forward_inference(noisy_mag, acc, vad)
    predict = (predict_mask * noisy_mag).cpu()
    predict = istft((predict.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").numpy()
    clean = clean.numpy().squeeze(1)
    return clean, predict
