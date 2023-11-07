import torch
from feature import stft, istft
from loss import Spectral_Loss, sisnr, StabilizedPermInvSISDRMetric
from model import batch_pesq
from evaluation import eval, AudioMetrics
import random

pit_loss = StabilizedPermInvSISDRMetric(n_actual_sources=2, n_estimated_sources=2, )
def enhancement_train(model, sample, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    return 0
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device)
    vad = sample['vad'].to(device)
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(noisy, 640, 320, 640)
    optimizer.zero_grad()
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
    loss = sisnr(est_audio, clean.squeeze(1))
    loss += Spectral_Loss(est_mag, clean_mag, vad)    
    if discriminator is not None:
        predict_fake_metric = discriminator(clean_mag, est_mag)
        loss += 0.05 * torch.nn.functional.mse_loss(predict_fake_metric.flatten(),
                                                torch.ones(noisy.shape[0]).to(device).float())
    loss.backward() 
    optimizer.step()
    if discriminator is not None:
        discrim_loss_metric = calculate_discriminator_loss(discriminator=discriminator, clean_mag=clean_mag, clean_audio=clean, est_mag = est_mag, est_audio=est_audio)
        if discrim_loss_metric is not None:
            optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])
    return loss.item()

def enhancement_test(model, sample, device='cuda'):
    return [0]
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
    vad = sample['vad'].to(device); dvector = sample['dvector'].to(device)
    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    clean_mag, _, _, _ = stft(clean, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")    
    est_audio = est_audio.cpu().numpy()
    clean = clean.cpu().numpy().squeeze(1)
    return eval(clean, est_audio)

def separation_train(model, sample, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
    mixture = sample['mixture'].to(device)

    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    batch, channel, time, freq = est_mag.shape
    est_audio = istft((est_mag.reshape(-1, time, freq), noisy_phase.repeat(1, 2, 1, 1).reshape(-1, time, freq)), 640, 320, 640, input_type="mag_phase")
    loss = pit_loss(est_audio.reshape(batch, channel, -1), mixture)
    loss.backward()
    optimizer.step()
    return loss.item()

def separation_test(model, sample, device='cuda'):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
    mixture = sample['mixture'].to(device)

    noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    batch, channel, time, freq = est_mag.shape
    est_audio = istft((est_mag.reshape(-1, time, freq), noisy_phase.repeat(1, 2, 1, 1).reshape(-1, time, freq)), 640, 320, 640, input_type="mag_phase")
    loss = -pit_loss(est_audio.reshape(batch, channel, -1), mixture)
    return [loss.item()]

def superresolution_train(model, sample, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device)
    optimizer.zero_grad()
    hr_clean = model(clean)
    sc_loss, mag_loss = model.loss(hr_clean.squeeze(1), clean.squeeze(1))
    loss = sc_loss + mag_loss
    optimizer.step()
    return loss.item()

def superresolution_test(model, sample, device='cuda'):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device);
    lr_clean = clean[:, :, ::4]
    hr_clean = model(lr_clean)
    metric = AudioMetrics.batch_evaluation(clean.squeeze(1).cpu().numpy(), hr_clean.squeeze(1).cpu().numpy())
    return metric

def identification_train(model, sample, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device)
    optimizer.zero_grad()
    acc_mag, _, _, _ = stft(acc, 640, 320, 640)
    embedding = model(acc_mag)
    loss = model.loss(embedding)
    loss.backward() 
    optimizer.step()
    return NotImplementedError

def identification_test(model, sample, device='cuda'):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
    vad = sample['vad'].to(device); dvector = sample['dvector'].to(device)
    acc_mag, _, _, _ = stft(acc, 640, 320, 640)

    enrollment_batch, verification_batch = torch.split(acc_mag, acc_mag.shape(0)//2, dim=0)
            
    enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3)))
    verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3)))
            
    perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
    unperm = list(perm)
    for i,j in enumerate(perm):
        unperm[j] = i
        
    verification_batch = verification_batch[perm]
    enrollment_embeddings = model(enrollment_batch)
    verification_embeddings = model(verification_batch)
    verification_embeddings = verification_embeddings[unperm]
    
    enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
    verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))
    
    enrollment_centroids = get_centroids(enrollment_embeddings)
    
    sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
    
    # calculating EER
    diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
    
    for thres in [0.01*i+0.5 for i in range(50)]:
        sim_matrix_thresh = sim_matrix>thres
        
        FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
        /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)

        FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
        /(float(hp.test.M/2))/hp.test.N)
        
        # Save threshold when FAR = FRR (=EER)
        if diff> abs(FAR-FRR):
            diff = abs(FAR-FRR)
            EER = (FAR+FRR)/2
            EER_thresh = thres
            EER_FAR = FAR
            EER_FRR = FRR
    batch_avg_EER += EER
    return NotImplementedError

def calculate_discriminator_loss(discriminator, clean_mag, clean_audio, est_mag, est_audio, ):
    pesq_score = batch_pesq(clean_audio.squeeze(1).cpu().numpy(), est_audio.detach().cpu().numpy())
    # The calculation of PESQ can be None due to silent part
    if pesq_score is not None:
        predict_enhance_metric = discriminator(
            clean_mag, est_mag.detach()
        )
        predict_max_metric = discriminator(
           clean_mag, clean_mag
        )
        discrim_loss_metric = torch.nn.functional.mse_loss(
            predict_max_metric.flatten(),  torch.ones(est_mag.shape[0]).to(est_mag.device))
        + torch.nn.functional.mse_loss(predict_enhance_metric.flatten(), pesq_score)
    else:
        discrim_loss_metric = None
    return discrim_loss_metric
