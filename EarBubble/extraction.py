from torch import optim
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet, ConvTasNet, SuDORMRFNet
from utils.models.beamforming.tfgridnet_realtime_embed.net import Net

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_neg_sisdr, multisrc_neg_sisdr
from utils.dataset.target_speech_dataset import MixtureDataset

import torch
import pytorch_lightning as pl
import numpy as np
import time
MOS_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to('cuda')

class SeparationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(SeparationLightningModule, self).__init__()
        self.config = config
        if self.config['output_format'] == 'separation':
            self.model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k')
            # self.model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k')
            # self.model = ConvTasNet(n_src=config['max_sources'], sample_rate=config['sample_rate'])
            # self.model = SuDORMRFNet(n_src=config['max_sources'], sample_rate=config['sample_rate'])
            self.loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
            print("permutation-invariant training")
        elif self.config['output_format'] in ['extraction', 'filtering', 'relay']:
            if self.config['output_format'] == 'relay':
                embed_dim = 194 # chunk + pad + 2
            else:
                embed_dim = 256
            model_params = {
                    "embed_dim": embed_dim,
                    "stft_chunk_size": 128,
                    "stft_pad_size": 64,
                    "num_ch": 1,
                    "num_src": 1,
                    "D": 64,
                    "L": 4,
                    "I": 1,
                    "J": 1,
                    "B": 3,
                    "H": 64,
                    "local_atten_len": 50,
                    "use_attn": True,
                    "lookahead": True,
                    "chunk_causal": True
                }
            self.model = Net(**model_params)
            self.loss = singlesrc_neg_sisdr
            print("single-source training")

    def training_step(self, batch, batch_idx):
        if self.config['output_format'] == 'separation':
            data, label = batch['mixture'], batch['sources']
            outputs = self.model(data)    
            loss = self.loss(outputs, label)
        elif self.config['output_format'] == 'extraction' or self.config['output_format'] == 'filtering':   
            data, embedding, label = batch['mixture'], batch['embeddings'], batch['sources']
            outputs = self.model((data, embedding))
            loss = self.loss(outputs.squeeze(), label.squeeze()).mean()
        else: # relay
            data, relay, label = batch['mixture'], batch['relay'], batch['sources']
            outputs = self.model((data, relay))
            loss = self.loss(outputs.squeeze(), label.squeeze()).mean()
        
        self.log('train', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config['output_format'] == 'separation':
            data, label = batch['mixture'], batch['sources']
            outputs = self.model(data)
            mixture = data.unsqueeze(1).repeat(1, self.config['max_sources'], 1)
            mixture_loss = self.loss(mixture, label)
            loss = self.loss(outputs, label)
            loss_i = loss - mixture_loss
        elif self.config['output_format'] == 'extraction' or self.config['output_format'] == 'filtering':   
            data, embedding, label = batch['mixture'], batch['embeddings'], batch['sources']
            outputs = self.model((data, embedding))
            mixture_loss = self.loss(data.squeeze(), label.squeeze()).mean()
            loss = self.loss(outputs.squeeze(), label.squeeze()).mean()
            loss_i = loss - mixture_loss
        else: # relay
            data, relay, label = batch['mixture'], batch['relay'], batch['sources']
            outputs = self.model((data, relay))
            mixture_loss = self.loss(data.squeeze(), label.squeeze()).mean()
            loss = self.loss(outputs.squeeze(), label.squeeze()).mean()
            loss_i = loss - mixture_loss
        self.log('val/snr', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)    
        self.log('val/snr_i', loss_i, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  

    def test_step(self, batch, batch_idx):
        '''
        We can store/ analyze/ visualize the output of the model here.
        '''
        # save the output to a folder of the logger
        import os
        import soundfile as sf
        save_folder = self.logger.log_dir + '/test_output'
        os.makedirs(save_folder, exist_ok=True)
        t_start = time.time()
        if self.config['output_format'] == 'relay':
            data, relay, label = batch['mixture'], batch['relay'], batch['sources'] 
            outputs = self.model((data, relay))
            outputs /= outputs.max()
        elif self.config['output_format'] == 'extraction' or self.config['output_format'] == 'filtering':   
            data, embedding, label = batch['mixture'], batch['embeddings'], batch['sources']
            outputs = self.model((data, embedding))
            outputs /= outputs.max()
        else:
            raise NotImplementedError("Test step is not implemented for this output format.")
        t_end = time.time()
        sample_folder = save_folder + f'/{batch_idx}'; os.makedirs(sample_folder, exist_ok=True)
        # save the output to a folder
        sf.write(os.path.join(sample_folder, f'output.mp3'), outputs.squeeze().cpu().numpy(), self.config['sample_rate'])
        sf.write(os.path.join(sample_folder, f'input.mp3'), data.squeeze().cpu().numpy(), self.config['sample_rate'])
        sf.write(os.path.join(sample_folder, f'label.mp3'), label.squeeze().cpu().numpy(), self.config['sample_rate'])
        mixture_loss = self.loss(data[0], label[0]).mean()
        loss = self.loss(outputs[0], label[0]).mean()
        loss_i = loss - mixture_loss
        # MOS predictor
        score = MOS_predictor(outputs[0], self.config['sample_rate'])
        self.log('test/snr', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)    
        self.log('test/snr_i', loss_i, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test/mos', score, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) 
        self.log('test/time', t_end - t_start, on_epoch=True, prog_bar=True, logger=True, sync_dist=True) 
        
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)
  

if __name__ == '__main__':
    
    config = { 
                "train_datafolder": 'dataset/MixLibriSpeech/LibriSpeech/train-clean-100',
                "test_datafolder": 'dataset/MixLibriSpeech/LibriSpeech/test-clean',

                "train_emb": 'dataset/MixLibriSpeech/librispeech_dvector_embeddings/train-clean-100',
                "test_emb": 'dataset/MixLibriSpeech/librispeech_dvector_embeddings/test-clean',

                "train_noise": 'dataset/MixLibriSpeech/wham_noise/tr', # None
                "test_noise": 'dataset/MixLibriSpeech/wham_noise/tt', # None

                "epochs": 20,
                "batch_size": 4,
                "duration": 5,
                "device": "cuda",
                "output_format": "relay",
                "sample_rate": 16000,
                "num_speakers": 2,
                "snr": [-10, 10],
                "relay_config":{
                    # 'MP3CompressorPerturb': {"vbr_min": 1, "vbr_max": 9.5},
                    # 'AACConversionPerturb': {'compress_rate_min': 4, 'compress_rate_max': 256},
                    'OPUSCodecsPerturb': {'compress_rate_min': 4, 'compress_rate_max': 4},
                    'BitCrushPerturb': {"bit_min": 4, "bit_max": 32},
                    'PacketLossPerturb': {"loss_rate_min": 0, "loss_rate_max": 0, "frame_time_min": 0.008, "frame_time_max": 0.05, 
                                          "decay_rate_min": 0, "decay_rate_max": 0.2, "hard_loss_prob": 1.0},
                    'WhiteNoisePerturb': {"snr_min": 5, "snr_max": 15},
                    # 'LatencyPerturb': {"min_delay_ms": 10, "max_delay_ms": 10}, # will ruin the SNR value
                },
                "mode": "microbenchmark3", # train, test, microbenchmark1, microbenchmark2, microbenchmark3
            }

    train_dataset = MixtureDataset(config['train_datafolder'], config['train_noise'], config['train_emb'], num_speakers=config["num_speakers"], 
                                   sample_rate=config['sample_rate'], duration=config['duration'], snr=config['snr'], relay_config=config['relay_config'],
                                     output_format=config['output_format'])
    test_dataset = MixtureDataset(config['test_datafolder'], config['test_noise'], config['test_emb'], num_speakers=config["num_speakers"],
                                    sample_rate=config['sample_rate'], duration=config['duration'], snr=config['snr'], relay_config=config['relay_config'],
                                    output_format=config['output_format'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = SeparationLightningModule(config)
    # ckpt = torch.load('runs/separation/extraction/checkpoints/epoch=19-step=71360.ckpt',weights_only=True)
    # ckpt = torch.load('runs/separation/relay_embedding/checkpoints/epoch=19-step=142700.ckpt',weights_only=True)
    # ckpt = torch.load('runs/separation/relay_mp3/checkpoints/epoch=3-step=14272.ckpt',weights_only=True)
    ckpt = torch.load('runs/separation/relay_opus/checkpoints/epoch=19-step=71360.ckpt', weights_only=True)
    model.load_state_dict(ckpt['state_dict'])

    logger = TensorBoardLogger("runs", name="separation")
    # trainer = Trainer(max_epochs=config['epochs'], devices=[0], logger=logger)
    trainer = Trainer(max_epochs=config['epochs'], logger=logger, accelerator='cpu')
    if config['mode'] == "train":
        trainer.fit(model, train_loader, test_loader)
    elif config['mode'] == "test":
        trainer.validate(model, test_loader)
    elif config['mode'] == "microbenchmark1": # only control the input snr
        import pandas as pd
        micro_benchmark = [] 
        # Do microbenchmark 1
        input_snrs = np.linspace(-10, 10, 10)
        for input_snr in input_snrs:
            config['snr'] = [input_snr, input_snr]
            test_dataset = MixtureDataset(config['test_datafolder'], config['test_noise'], config['test_emb'], num_speakers=config["num_speakers"],
                                        sample_rate=config['sample_rate'], duration=config['duration'], snr=config['snr'], relay_config=config['relay_config'],
                                        output_format=config['output_format'])
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
            result = trainer.validate(model, test_loader)[0]
            snr = result['val/snr']; snr_i = result['val/snr_i']
            micro_benchmark += [[input_snr, snr, snr_i]]
        # add the column names
        columns = ['input_snr', 'snr', 'snr_i']
        # create a dataframe
        df = pd.DataFrame(micro_benchmark, columns=columns)
        # save the dataframe to a csv file  
        df.to_csv('micro_benchmark.csv', index=False)
    elif config['mode'] == "microbenchmark2": # control the input snr and relay audio
        import pandas as pd
        micro_benchmark = []
        input_snrs = np.linspace(-10, 10, 20)
        factors = np.linspace(1, 9, 9); factor_name = 'MP3CompressorPerturb'
        factors = [2, 4, 8, 16, 32, 64, 128, 256]; factor_name = 'OPUSCodecsPerturb'
        for input_snr in input_snrs:
            # modify the config to set the SNR
            config['snr'] = [input_snr, input_snr]
            for factor in factors:
                # modify the config to set the MP3CompressorPerturb, need modification for other perturbations
                # config['relay_config']['MP3CompressorPerturb'] = {"vbr_min": factor, "vbr_max": factor}
                config['relay_config']['OPUSCodecsPerturb'] = {'compress_rate_min': factor, 'compress_rate_max': factor}
                # create the dataset and dataloader
                test_dataset = MixtureDataset(config['test_datafolder'], config['test_noise'], config['test_emb'], num_speakers=config["num_speakers"], 
                                        sample_rate=config['sample_rate'], duration=config['duration'], snr=config['snr'], relay_config=config['relay_config'],
                                        output_format=config['output_format'])
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
                result = trainer.validate(model, test_loader)[0]
                snr = result['val/snr']; snr_i = result['val/snr_i']
                print(f"input_snr: {input_snr}, factor: {factor}, snr: {snr}, snr_i: {snr_i}")
                micro_benchmark += [[input_snr, factor, snr, snr_i]]
        # add the column names
        columns = ['input_snr', factor_name, 'snr', 'snr_i']
        # create a dataframe
        df = pd.DataFrame(micro_benchmark, columns=columns)
        # save the dataframe to a csv file
        df.to_csv('micro_benchmark.csv', index=False)
    elif config['mode'] == "microbenchmark3": # measure the latency compensation
        config['relay_config']['LatencyPerturb'] = {"min_delay_ms": 0, "max_delay_ms": 0}
        test_dataset = MixtureDataset(config['test_datafolder'], config['test_noise'], config['test_emb'], num_speakers=config["num_speakers"],
                                        sample_rate=config['sample_rate'], duration=config['duration'], snr=config['snr'], relay_config=config['relay_config'],
                                        output_format=config['output_format'])
        # only the first 10 samples to save time/ storage
        test_dataset.mixture_files = test_dataset.mixture_files[:10]
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
        trainer.test(model, test_loader)
