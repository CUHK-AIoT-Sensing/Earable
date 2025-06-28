from torch import optim
# We train the same model architecture that we used for inference above.
from models import SNR_Estimator_LSTM, SNR_Estimator_CNN
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import singlesrc_neg_sisdr
import pytorch_lightning as pl
import soundfile as sf
import os
from utils.vib_dataset import ABCS_dataset, EMSB_dataset, V2S_dataset
from utils.base_dataset import SpeechEnhancementDataset
import torch


class SNRESTIMATOR_LightningModule(pl.LightningModule):
    def __init__(self, config):
        super(SNRESTIMATOR_LightningModule, self).__init__()
        self.config = config
        if config['model'] == 'lstm':
            self.model = SNR_Estimator_LSTM(**config["model"]["params"])
        else:
            self.model = SNR_Estimator_CNN(**config["model"]["params"])
        self.loss = torch.nn.MSELoss()  # Using MSE loss for SNR estimation
        # save the config as hparams.yaml
        self.save_hyperparameters(config)

    def normalize_snr(self, snr):
        """Normalize SNR to a range of 0 to 1."""
        min_snr = self.config['snr_range'][0]
        max_snr = self.config['snr_range'][1]
        return (snr - min_snr) / (max_snr - min_snr)

    def training_step(self, batch, batch_idx):
        input_data = [batch[key] for key in self.config['input']]
        ref_output = batch[self.config['output']]  # Assuming single output for simplicity
        ref_output = self.normalize_snr(ref_output)  # Normalize SNR if needed
        outputs = self.model(*input_data) 
        outputs = outputs.squeeze(1)
        loss = self.loss(outputs, ref_output).mean()
        self.log('train', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data = [batch[key] for key in self.config['input']]
        ref_output = batch[self.config['output']]  # Assuming single output for simplicity
        ref_output = self.normalize_snr(ref_output)  # Normalize SNR if needed
        outputs = self.model(*input_data) 
        outputs = outputs.squeeze(1)
        loss = torch.abs(outputs - ref_output).mean() * (self.config['snr_range'][1] - self.config['snr_range'][0])  # Scale loss to original SNR range
        self.log('val/mae', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
    
def dataset_parser(dataset_name, split='all'):
    if dataset_name == 'ABCS':
        dataset = ABCS_dataset(split=split)
    elif dataset_name == 'EMSB':
        dataset = EMSB_dataset(split=split)
    elif dataset_name == 'V2S':
        dataset = V2S_dataset(split=split)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset

if __name__ == "__main__":
    # Example usage
    import json
    config_name = 'snr_estimator'
    config = json.load(open(f"config/{config_name}.json", "r"))
    vib_dataset = dataset_parser(config['dataset'], config['split'])

    dataset = SpeechEnhancementDataset(dataset=vib_dataset, noise_folders=config['noise_dataset'], rir=config['rir'],
                                       length=config['length'], sample_rate=config['sample_rate'], snr_range=config['snr_range'], )
    train_dataset = torch.utils.data.Subset(dataset, range(int(len(dataset) * 0.8)))
    val_dataset = torch.utils.data.Subset(dataset, range(int(len(dataset) * 0.8), len(dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    logger = TensorBoardLogger("runs", name=config_name)
    trainer = Trainer(max_epochs=config['epochs'], logger=logger, accelerator='gpu', devices=[0])
    if config['checkpoint']:
        print(f"Loading checkpoint from {config['checkpoint']}")
        model = SNRESTIMATOR_LightningModule.load_from_checkpoint(config['checkpoint'], config=config)
        trainer.validate(model, val_loader)
    else:
        model = SNRESTIMATOR_LightningModule(config)
        trainer.fit(model, train_loader, val_loader)