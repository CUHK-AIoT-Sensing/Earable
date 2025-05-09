# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
import pytorch_lightning as pl
# We train the same model architecture that we used for inference above.

from utils.models.beamforming import Net, EmbedTFGridNet, Net_embed
from utils.models.adaptive_loss import SNRLPLoss
from asteroid.losses.sdr import SingleSrcNegSDR, MultiSrcNegSDR


from utils.dataset.beamforming_dataset import Beamforming_dataset
import torch.nn as nn
import torch    
import numpy as np

class CosineSimilarityLoss(nn.Module):
    '''
    Input: (B, C)
    '''
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)
    def forward(self, x, y):
        cosine_sim = self.cos(x, y)
        return 1 - cosine_sim.mean()
       

class BeamformingLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(BeamformingLightningModule, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        if self.config['output_format'] == 'dvector_distill':
            self.model = EmbedTFGridNet(embed_dim=256, num_ch=config['num_channel'], n_fft=128, stride=64, num_blocks=3)
            self.loss = CosineSimilarityLoss()
        elif self.config['output_format'] == 'dvector_filter':
            self.model = Net_embed(embed_dim=256, num_ch=1, num_src=1)
            self.loss = MultiSrcNegSDR('sisdr')
        elif self.config['output_format'] == 'beamforming':
            self.model = Net(num_ch=config['num_channel'], num_src=1)
            self.loss = MultiSrcNegSDR('sisdr')
        elif self.config['output_format'] == 'region':
            self.model = Net(num_ch=config['num_channel'], num_src=config['num_region'])
            self.loss = SNRLPLoss()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        outputs = self.model(data)

        if self.config['output_format'] == 'region':
            if self.current_epoch < 15:
                loss = self.loss(outputs, label, neg_weight=0)
            else:
                loss = self.loss(outputs, label, neg_weight=20)
        else:
            loss = self.loss(outputs, label).mean()
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        outputs = self.model(data)
        if self.config['output_format'] == 'region':
            positive_loss, negative_loss = self.loss(outputs, label, neg_weight=1)
            self.log('validataion/positive', positive_loss, on_epoch=True, prog_bar=True, logger=True)
            self.log('validataion/negative', negative_loss, on_epoch=True, prog_bar=True, logger=True)
        else:
            loss = self.loss(outputs, label).mean()
            self.log('validataion/loss', loss, on_epoch=True, prog_bar=True, logger=True)
           
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
    
    def visualize(self, test_dataloader):
        assert self.config['output_format'] != 'dvector' # output is not dvector
        import matplotlib.pyplot as plt
        import soundfile as sf
        self.eval()
        # sample random batch
        batch = next(iter(test_dataloader))
        data, label = batch
        outputs = self(data)
        loss = self.loss(outputs, label, neg_weight=1)
        B, C, T = label.shape
        for b in range(B):
            label_sample = label[b]; outputs_sample = outputs[b] # (N_channel, T), (1, T), (1, T)             
            max_value = max(label_sample.max(), outputs_sample.max()).item()
            fig, axs = plt.subplots(C, 2, figsize=(10, 10))
            for i in range(C):
                axs[i, 0].plot(label_sample[i, :].numpy(), c='b')
                axs[i, 1].plot(outputs_sample[i, :].detach().numpy(), c='g')
                axs[i, 0].set_ylim(-max_value, max_value)
                axs[i, 1].set_ylim(-max_value, max_value)
            plt.savefig(f'./utils/resources/beamforming_vis_{b}.png')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = { 
                "train_datafolder": "dataset/simulation/smartglass_TIMIT_event_2/train",
                "test_datafolder": "dataset/simulation/smartglass_TIMIT_event_2/test",
                "ckpt": "",
                "duration": 5,
                "epochs": 20,
                "batch_size": 4,
                "output_format": "dvector_filter",
                "sample_rate": 16000,
                "max_sources": 2,
                "num_channel": 5, 
                "num_region": 4,
            }
    train_dataset = Beamforming_dataset(config['train_datafolder'], config,)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    test_dataset = Beamforming_dataset(config['test_datafolder'], config,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    model = BeamformingLightningModule(config)

    logger = pl.loggers.TensorBoardLogger('runs', name='beamforming/' + config['output_format'])
    trainer = Trainer(max_epochs=config['epochs'], devices=[1], logger=logger)
    
    # ckeckpoint = 'runs/beamforming/dvector/version_0/checkpoints/epoch=19-step=25740.ckpt' 
    # ckeckpoint = 'runs/beamforming/dvector_distill/version_1/checkpoints/epoch=19-step=25700.ckpt' 
    # model.load_state_dict(torch.load(ckeckpoint, weights_only=True)['state_dict'])
    trainer.fit(model, train_loader, test_loader)  
    # model.visualize(test_loader)
    #trainer.validate(model, test_loader)

