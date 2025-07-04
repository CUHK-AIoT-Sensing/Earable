import torch
from utils.base_dataset import SpeechEnhancementDataset
from speech_enhancement import SpeechEnhancementLightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import json
from utils.vib_dataset import ABCS_dataset, EMSB_dataset, V2S_dataset
from utils.bcf_dataset import BCFAugmentationDataset, aishellDataset
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
def dataset_parser(dataset_name, split='all'):
    if dataset_name == 'ABCS':
        dataset = ABCS_dataset(split=split)
    elif dataset_name == 'EMSB':
        dataset = EMSB_dataset(split=split)
    elif dataset_name == 'V2S':
        dataset = V2S_dataset(split=split)
    elif dataset_name == 'aishell':
        audio_dataset = aishellDataset(folder='../dataset/Audio/aishell', split=split)
        dataset = BCFAugmentationDataset(audio_dataset, bcf_dataset='ABCS')
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Enhancement Training and Validation")
    parser.add_argument('--config', type=str, default='tfgridnet_embed_adaptive', help='Configuration name for the experiment')
    args = parser.parse_args()

    config_name = args.config
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

    from main_e2e import snr_wise_validation
    config['model']['params']['B'] = 4
    if not config['checkpoint_path']:
        model = SpeechEnhancementLightningModule(config=config)
    else:
        model = SpeechEnhancementLightningModule.load_from_checkpoint(config['checkpoint_path'], config=config)
        print(f"Loaded model from {config['checkpoint_path']}")
    
    trainer.fit(model, train_loader, val_loader)
    snr_wise_validation(config, config_name + '_4', vib_dataset, trainer, model)
