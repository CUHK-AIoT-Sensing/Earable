import torch
from feature import stft, istft
from loss import get_loss, eval
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import copy
def train_epoch(model, train_loader, optimizer, device='cuda', discriminator=None, optimizer_disc=None):
    Loss_list = []
    model.train()
    pbar = tqdm(train_loader)
    for sample in pbar:
        acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device)
        vad = sample['vad'].to(device)
        noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
        clean_mag, _, _, _ = stft(noisy, 640, 320, 640)
        optimizer.zero_grad()
        acc, _, _, _ = stft(acc, 640, 320, 640)
        est_mag = model(noisy_mag, acc)
        est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
        loss = get_loss(est_audio, clean.squeeze(1))
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
        Loss_list.append(loss.item())
        pbar.set_description("loss: %.3f" % (loss.item()))
    return np.mean(Loss_list)

def test_epoch(model, dataset, BATCH_SIZE, device='cuda'):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    Metric = []
    model.eval()
    with torch.no_grad():
        for sample in pbar:
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
            vad = sample['vad'].to(device)
            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            clean_mag, _, _, _ = stft(clean, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            est_mag = model(noisy_mag, acc)
            est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")    
            metric = eval(clean.squeeze(1), est_audio)
            Metric.append(metric)  
    avg_metric = np.round(np.mean(np.concatenate(Metric, axis=0), axis=0),2).tolist()
    print(avg_metric)
    return avg_metric

def test_epoch_save(model, dataset, BATCH_SIZE, dir, output_dir, device='cuda'):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=True)
    pbar = tqdm(test_loader)
    model.eval()
    with torch.no_grad():
        for sample in pbar:
            acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
            vad = sample['vad'].to(device)
            noisy_mag, noisy_phase, noisy_real, noisy_imag = stft(noisy, 640, 320, 640)
            clean_mag, _, _, _ = stft(clean, 640, 320, 640)
            acc, _, _, _ = stft(acc, 640, 320, 640)
            est_mag = model(noisy_mag, acc)
            est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")  
            for j in range(BATCH_SIZE):
                fname = sample['file'][j]
                fname = fname.replace(dir, output_dir)
                wavfile.write(fname, 16000, est_audio[j])

def RemixIT(i, sample, optimizer, model, teacher_model, device='cuda', t_momentum=0.95):
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device)
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    acc, _, _, _ = stft(acc, 640, 320, 640)
    with torch.no_grad():
        est_mag = teacher_model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").unsqueeze(1)
    noise = noisy - est_audio

    sample['noise'] = noise.detach().cpu()
    sample['clean'] = est_audio.detach().cpu()
    sample['noisy'] = sample['clean'] + sample['noise'][torch.randperm(sample['noise'].shape[0])]
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    optimizer.zero_grad()
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
    loss = get_loss(est_audio, clean.squeeze(1))
    loss.backward() 
    optimizer.step()
    if i % 8 == 0 and i > 0:
        new_teacher_w = copy.deepcopy(teacher_model.state_dict())
        student_w = model.state_dict()
        for key in new_teacher_w.keys():
            new_teacher_w[key] = (
                    t_momentum * new_teacher_w[key] + (1.0 - t_momentum) * student_w[key])

        teacher_model.load_state_dict(new_teacher_w)
        del new_teacher_w
        teacher_model.eval()
    return loss.item()
def PSE_DP(i, sample, optimizer, model, teacher_model, device='cuda', t_momentum=0.95):
    noisy = sample['noisy'].to(device)
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    with torch.no_grad():
        segsnr = teacher_model(noisy_mag, acc)

    # don't finish Data purification yet
    sample['clean'] = sample['noisy']
    sample['noisy'] += sample['noisy'][torch.randperm(sample['noisy'].shape[0])]
    acc = sample['imu'].to(device); noisy = sample['noisy'].to(device); clean = sample['clean'].to(device); 
    noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
    optimizer.zero_grad()
    acc, _, _, _ = stft(acc, 640, 320, 640)
    est_mag = model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase")
    loss = get_loss(est_audio, clean.squeeze(1), vad=segsnr)
    loss.backward() 
    optimizer.step()
    return loss.item()
def AlterNet_E(model, train_loader, optimizer, teacher_model, device='cuda'):
    teacher_model.train()
    model.eval()
    Loss_list = []
    pbar = tqdm(train_loader)
    for i, sample in enumerate(pbar):
        acc = sample['imu'].to(device); noisy = sample['noisy'].to(device)
        noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
        acc, _, _, _ = stft(acc, 640, 320, 640)
        with torch.no_grad():
            est_mag = model(noisy_mag, acc)
        optimizer.zero_grad()
        mask = teacher_model(acc)
        loss = torch.nn.functional.mse_loss(mask * noisy_mag, est_mag)
        Loss_list.append(loss)
        pbar.set_description("loss: %.3f" % (loss))
    return np.mean(Loss_list)
def AlterNet_M(model, train_loader, optimizer, teacher_model, device='cuda'):
    teacher_model.eval()
    model.train()
    Loss_list = []
    pbar = tqdm(train_loader)
    for i, sample in enumerate(pbar):
        acc = sample['imu'].to(device); noisy = sample['noisy'].to(device)
        noisy_mag, noisy_phase, _, _ = stft(noisy, 640, 320, 640)
        acc, _, _, _ = stft(acc, 640, 320, 640)
        with torch.no_grad():
            clean_mag = teacher_model(acc)
        optimizer.zero_grad()
        est_mag = model(noisy_mag, acc)
        loss = torch.nn.functional.mse_loss(est_mag, clean_mag)
        Loss_list.append(loss)
        pbar.set_description("loss: %.3f" % (loss))
    return np.mean(Loss_list)
    

def unsupervised_train_epoch(model, train_loader, optimizer, teacher_model, device='cuda'):
    teacher_model.eval()
    model.train()
    Loss_list = []
    pbar = tqdm(train_loader)
    for i, sample in enumerate(pbar):
        loss = RemixIT(i, sample, optimizer, model, teacher_model, device)
        # loss = PSE_DP(i, sample, optimizer, model, teacher_model, device)
        Loss_list.append(loss)
        pbar.set_description("loss: %.3f" % (loss))
    return np.mean(Loss_list)
import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score
def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy)
    )
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to("cuda")

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
