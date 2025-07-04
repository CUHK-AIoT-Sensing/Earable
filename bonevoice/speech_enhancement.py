import torch
from torch import optim
# We train the same model architecture that we used for inference above.
from asteroid.models import SuDORMRFNet, DPRNNTasNet
from models import VibVoice, Baseline, TFGridNetRealtimeEmbed

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import singlesrc_neg_sisdr
import pytorch_lightning as pl
import soundfile as sf
import os
from torchmetrics.audio import pesq

def Log_Spectral_Distance(y_true, y_pred):
    """
    Log-Spectral Distance: $LSD(x, y)= \frac{1}{L} \sum_{l=1}^{L} \sqrt{\frac{1}{K}\sum_{k=1}^{K}(X(l,k) - \hat{X}(l,k))^2}$,
    """
    spec_true = torch.stft(y_true, n_fft=512, hop_length=128, win_length=512, window=torch.hann_window(512), return_complex=True)
    spec_pred = torch.stft(y_pred, n_fft=512, hop_length=128, win_length=512, window=torch.hann_window(512), return_complex=True)
    log_spec_true = torch.log(spec_true ** 2); log_spec_pred = torch.log(spec_pred ** 2)
    LSD = torch.sqrt(torch.mean((log_spec_true - log_spec_pred) ** 2, dim=-1)).mean()
    return LSD


def model_parser(model_config):
    if model_config['name'] == 'SuDORMRFNet':
        return SuDORMRFNet(**model_config['params'])  
    elif model_config['name'] == 'DPRNNTasNet':
        return DPRNNTasNet(**model_config['params'])
    elif model_config['name'] == 'Baseline':
        return Baseline(**model_config['params'])
    elif model_config['name'] == 'VibVoice':
        return VibVoice(**model_config['params'])
    elif model_config['name'] == 'TFGridNetRealtimeEmbed':
        return TFGridNetRealtimeEmbed(**model_config['params'])
    else:
        raise ValueError(f"Unknown model name: {model_config['name']}")


class SpeechEnhancementLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(SpeechEnhancementLightningModule, self).__init__()
        self.config = config
        self.model = model_parser(config['model'])
        self.loss = singlesrc_neg_sisdr
        # save the config as hparams.yaml
        self.save_hyperparameters(config)
        # self.pesq = pesq.PerceptualEvaluationSpeechQuality(16000, 'wb')

    def training_step(self, batch, batch_idx):
        input_data = [batch[key] for key in self.config['input']]
        ref_output = batch[self.config['output']]  # Assuming single output for simplicity
        outputs = self.model(*input_data)    
        outputs = outputs.squeeze(1)
        loss = self.loss(outputs, ref_output).mean()
        self.log('train', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data = [batch[key] for key in self.config['input']]
        ref_output = batch[self.config['output']]  # Assuming single output for simplicity
        outputs = self.model(*input_data) 
        outputs = outputs.squeeze(1)
        loss = self.loss(outputs, ref_output).mean()
        mixture_loss = self.loss(batch['noisy_audio'], ref_output).mean()
        # self.log('val/mixture', mixture_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        # pesq_score = self.pesq(outputs, ref_output)

        self.log('val/sisnr', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/sisnr_i', loss - mixture_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        # self.log('val/pesq', pesq_score, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        

    def test_step(self, batch, batch_idx):
        input_data = [batch[key] for key in self.config['input']]
        ref_output = batch[self.config['output']]
        outputs = self.model(*input_data)
        loss = self.loss(outputs.squeeze(1), ref_output).mean()
        print(f"Test Loss: {loss.item()}")
        mixture_loss = self.loss(input_data[0].squeeze(1), ref_output).mean()
        print(f"Mixture Loss: {mixture_loss.item()}")

        os.makedirs(f"cache/{batch_idx}", exist_ok=True)
        sf.write(f"cache/{batch_idx}/output.wav", outputs.squeeze().cpu().numpy(), self.config['sample_rate'])
        sf.write(f"cache/{batch_idx}/reference.wav", ref_output.squeeze().cpu().numpy(), self.config['sample_rate'])
        sf.write(f"cache/{batch_idx}/mixture.wav", input_data[0].squeeze().cpu().numpy(), self.config['sample_rate'])
        if len(input_data) > 1:
            # Assuming the second input is vibration data
            sf.write(f"cache/{batch_idx}/vibration_input.wav", input_data[1].squeeze().cpu().numpy(), self.config['sample_rate'])

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
  