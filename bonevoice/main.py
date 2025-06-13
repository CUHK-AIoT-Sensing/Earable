import torch
from utils.base_dataset import SpeechEnhancementDataset
from speech_enhancement import SpeechEnhancementLightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import json
from utils.vib_dataset import ABCS_dataset, EMSB_dataset, V2S_dataset
from utils.bcf_dataset import BCFDataset, aishellDataset
import argparse
import pandas as pd
from tqdm import tqdm
def dataset_parser(dataset_name, split='all'):
    if dataset_name == 'ABCS':
        dataset = ABCS_dataset(split=split)
    elif dataset_name == 'EMSB':
        dataset = EMSB_dataset(split=split)
    elif dataset_name == 'V2S':
        dataset = V2S_dataset(split=split)
    elif dataset_name == 'aishell':
        audio_dataset = aishellDataset(folder='../dataset/Audio/aishell', split=split)
        dataset = BCFDataset(audio_dataset, bcf_dataset='ABCS')
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Enhancement Training and Validation")
    parser.add_argument('--config', type=str, default='tfgridnet_embed_bcf', help='Configuration name for the experiment')
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

    model = SpeechEnhancementLightningModule(config=config)
    logger = TensorBoardLogger("runs", name=config_name)
    trainer = Trainer(max_epochs=config['epochs'], logger=logger, accelerator='gpu', devices=[0])
    trainer.fit(model, train_loader, val_loader)
    
    # logger = TensorBoardLogger("runs", name=config_name)
    # trainer = Trainer(max_epochs=config['epochs'], logger=logger, accelerator='gpu', devices=[0])
    # # # checkpoint_path = f"runs/{config_name}/version_41/checkpoints/epoch=4-step=11505.ckpt"
    # # checkpoint_path = f"runs/{config_name}/version_6/checkpoints/epoch=4-step=11505.ckpt"
    # checkpoint_path = f"runs/{config_name}/version_1/checkpoints/epoch=4-step=23970.ckpt"
    # model = SpeechEnhancementLightningModule.load_from_checkpoint(checkpoint_path, config=config)

    ## Validation for each noise folder
    # results = {}
    # noise_folders = ["../dataset/Audio/aishell/aishell-dev", "../dataset/Audio/ESC50", "self", "../dataset/Audio/DEMAND"]
    # for noise_folder in noise_folders:
    #     print(f"Noise folder: {noise_folder}")
    #     dataset = SpeechEnhancementDataset(dataset=vib_dataset, noise_folders=[noise_folder], rir=config['rir'],
    #                                    length=config['length'], sample_rate=config['sample_rate'], snr_range=config['snr_range'], )
    #     val_dataset = torch.utils.data.Subset(dataset, range(int(len(dataset) * 0.8), len(dataset)))
    #     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    #     result = trainer.validate(model, val_loader)
    #     results[noise_folder] = result[0]
    # save results to a CSV file
    # results_df = pd.DataFrame(results).T
    # results_df.to_csv(f"{config_name}_results.csv", index_label='split')
    # print("Validation results saved to CSV file.")
    

    # # Validation for each split/ speaker
    # splits = vib_dataset.json_file.keys()
    # results = {}
    # for split in tqdm(splits):
    #     print(f"Validating split: {split}")
    #     val_dataset = dataset_parser(config['dataset'], split=[split])
    #     dataset = SpeechEnhancementDataset(dataset=val_dataset, noise_folders=config['noise_dataset'], rir=config['rir'],
    #                                        length=config['length'], sample_rate=config['sample_rate'], snr_range=config['snr_range'])
    #     val_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    #     result = trainer.validate(model, val_loader, verbose=False)
    #     results[split] = result[0]
    # # save results to a CSV file
    # results_df = pd.DataFrame(results).T
    # results_df.to_csv(f"{config_name}_results.csv", index_label='split')
    # print("Validation results saved to CSV file.")
    

  
        