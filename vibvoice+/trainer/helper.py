import torch
from feature import stft, istft
import copy

def concat_sample(sample1, sample2):
    key_list = ['imu', 'noisy', 'clean']
    for key in key_list:
        if key not in sample1.keys():
            sample1[key] = sample2[key]
        else:
            sample1[key] = torch.cat([sample1[key], sample2[key]], dim=0)
    if 'file' in sample1.keys():
        sample1['file'] += sample2['file']
    else:
        sample1['file'] = sample2['file']
    return sample1
def pad_sample(sample, pad_len, num_batch):
    key_list = ['imu', 'noisy', 'clean', ]
    for key in key_list: 
        sample[key] = torch.nn.functional.pad(sample[key], (0, pad_len - sample[key].shape[-1], 0, 0))
        sample[key] = sample[key].reshape(num_batch, 1, -1)
    return sample
def batching(sample, batch, batch_count, length_count, length=10, sr=16000):
    len_recording = sample['noisy'].shape[-1]
    num_batch = int(len_recording / (sr * length)) + 1
    padded_sample = pad_sample(sample, num_batch * sr * length, num_batch)
    batch = concat_sample(batch, padded_sample)
    return batch, batch_count + [num_batch], length_count + [len_recording]
def unbatching(batch, batch_count, length_count):
    list_sample = []
    count = 0
    for batch_c, length_c in zip(batch_count, length_count):
        sample = batch[count: count+batch_c, :]
        sample = sample.reshape(1, -1)[:, :length_c]
        count += batch_c
        list_sample.append(sample)
    return list_sample

def data_purification(noisy_mag, acc, sample):
    noisy_mag = torch.mean(noisy_mag, dim = -1,)
    acc = torch.mean(acc, dim = -1,)
    acc *= torch.sum(noisy_mag, dim = -1, keepdim=True) / torch.sum(acc, dim = -1, keepdim=True)
    segsnr = torch.log10(acc / (noisy_mag + 1e-8) + 1e-8)
    segsnr = torch.clip(segsnr, min=-10, max=35).unsqueeze(-1)
    segsnr = (segsnr + 10) / 45 

    sample['clean'] = sample['noisy']
    sample['noisy'] += sample['noisy'][torch.randperm(sample['noisy'].shape[0])]
    return sample, segsnr


def Remix(noisy_mag, acc, noisy_phase, noisy, teacher_model, sample):
    with torch.no_grad():
        est_mag = teacher_model(noisy_mag, acc)
    est_audio = istft((est_mag.squeeze(1), noisy_phase.squeeze(1)), 640, 320, 640, input_type="mag_phase").unsqueeze(1)
    noise = noisy - est_audio

    sample['noise'] = noise
    sample['clean'] = est_audio
    sample['noisy'] = sample['clean'] + sample['noise'][torch.randperm(sample['noise'].shape[0])]
    return sample
    
def update_teacher(model, teacher_model, t_momentum=0.99):
    new_teacher_w = copy.deepcopy(teacher_model.state_dict())
    student_w = model.state_dict()
    for key in new_teacher_w.keys():
        new_teacher_w[key] = (
                t_momentum * new_teacher_w[key] + (1.0 - t_momentum) * student_w[key])

    teacher_model.load_state_dict(new_teacher_w)
    del new_teacher_w
    teacher_model.eval()
    return teacher_model

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
            mask = teacher_model(acc)
        optimizer.zero_grad()
        est_mag = model(noisy_mag, acc)
        loss = torch.nn.functional.mse_loss(est_mag, mask * noisy_mag)
        Loss_list.append(loss)
        pbar.set_description("loss: %.3f" % (loss))
    return np.mean(Loss_list)
    
