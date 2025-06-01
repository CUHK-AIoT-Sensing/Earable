import torch
from utils.base_dataset import SpeechEnhancementDataset
from speech_enhancement import SpeechEnhancementLightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import json

if __name__ == "__main__":
    config = json.load(open("config/tfgridnet.json", "r"))
    dataset = SpeechEnhancementDataset(dataset_names=config['dataset'], noises_folders=config['noise_dataset'], 
                                       length=config['length'], sample_rate=config['sample_rate'], snr_range=config['snr_range'])

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    model = SpeechEnhancementLightningModule(config=config)
    logger = TensorBoardLogger("runs", name="SE")
    trainer = Trainer(max_epochs=config['epochs'], logger=logger, accelerator='gpu', devices=[0, 1])
    trainer.fit(model, train_loader, test_loader)

    