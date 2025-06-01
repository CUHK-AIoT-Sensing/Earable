from torch import optim
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# We train the same model architecture that we used for inference above.
from asteroid.models import SuDORMRFNet
from models import VibVoice, VibVoice_Early, TFGridNetRealtime, Multimodal_TFGridNetRealtime

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_neg_sisdr, multisrc_neg_sisdr
import pytorch_lightning as pl

def model_parser(model_config):
    if model_config['name'] == 'SuDORMRFNet':
        return SuDORMRFNet(**model_config['params'])  
    elif model_config['name'] == 'VibVoice':
        return VibVoice(**model_config['params'])
    elif model_config['name'] == 'TFGridNetRealtime':
        return TFGridNetRealtime(**model_config['params'])
    elif model_config['name'] == 'Multimodal_TFGridNetRealtime':
        return Multimodal_TFGridNetRealtime(**model_config['params'])


class SpeechEnhancementLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(SpeechEnhancementLightningModule, self).__init__()
        self.config = config
        # self.model = ConvTasNet(n_src=config['max_sources'], sample_rate=config['sample_rate'])
        self.model = model_parser(config['model'])
        self.loss = singlesrc_neg_sisdr

    def training_step(self, batch, batch_idx):
        input_data = [batch[key] for key in self.config['input']]
        if len(input_data) == 1:
            input_data = input_data[0]
        ref_output = batch[self.config['output']]  # Assuming single output for simplicity
        outputs = self.model(input_data)    
        outputs = outputs.squeeze(1)
        loss = self.loss(outputs, ref_output).mean()
        self.log('train', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data = [batch[key] for key in self.config['input']]
        if len(input_data) == 1:
            input_data = input_data[0]
        ref_output = batch[self.config['output']]  # Assuming single output for simplicity
        outputs = self.model(input_data) 
        outputs = outputs.squeeze(1)
        loss = -self.loss(outputs, ref_output).mean()
        mixture_loss = -self.loss(input_data, ref_output).mean()
        # self.log('val/sisnr', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/sisnr_i', loss - mixture_loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)
  