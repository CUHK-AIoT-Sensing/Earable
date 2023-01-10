import os
import random
import librosa

import numpy as np
import pandas as pd

import torch.utils.data as td
from typing import Optional
import torchvision as tv
class ToTensor1D(tv.transforms.ToTensor):
    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])

        return tensor_2d.squeeze_(0)
class FSD50K(td.Dataset):
    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 fold: Optional[int] = None,
                 transform_audio=ToTensor1D(),
                 few_shot=None,
                 length=None,
                 **_):

        super(FSD50K, self).__init__()

        self.sample_rate = sample_rate
        self.train = train
        if self.train:
            meta = self.load_meta(os.path.join(root, 'FSD.ground_truth', 'dev.csv'))
        else:
            meta = self.load_meta(os.path.join(root, 'FSD.ground_truth', 'eval.csv'))
        if fold is None:
            fold = 5

        self.folds_to_load = set(meta['fold'])

        if fold not in self.folds_to_load:
            raise ValueError(f'fold {fold} does not exist')

        self.transform = transform_audio



        self.data = list()
        self.load_data(meta, os.path.join(root, 'audio'), few_shot)
        self.class_idx_to_label = dict()
        for row in self.data:
            idx = row['target']
            label = row['category']
            self.class_idx_to_label[idx] = label
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}
        self.length = length

    @staticmethod
    def load_meta(path_to_csv: str) -> pd.DataFrame:
        meta = pd.read_csv(path_to_csv)
        return meta

    def load_data(self, meta: pd.DataFrame, base_path: str, few_shot=None):
        class_count = dict()
        for idx, row in meta.iterrows():
            if row['fold'] in self.folds_to_load:
                if few_shot is None:
                    pass
                elif row['target'] in class_count:
                    if class_count[row['target']] >= few_shot:
                        continue
                self.data.append({
                    'audio': os.path.join(base_path, row['filename']),
                    'sample_rate': self.sample_rate,
                    'target': row['target'],
                    'category': row['category'].replace('_', ' '),
                    'fold': row['fold'],
                    'esc10': row['esc10']
                })
                if row['target'] in class_count:
                    class_count[row['target']] += 1
                else:
                    class_count[row['target']] = 1

    def __getitem__(self, index: int):
        if not (0 <= index < len(self)):
            raise IndexError

        sample = self.data[index]
        filename: str = sample['audio']
        audio, sample_rate = librosa.load(filename, sr=sample['sample_rate'], mono=True)
        t_start = random.sample(range(len(audio) - self.length * sample_rate + 1), 1)[0]
        audio = audio[t_start: t_start + self.length * sample_rate]
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        audio = (audio.T * 32768.0).astype(np.float32)
        target = [self.data[index]['category']]

        if self.transform is not None:
            audio = self.transform(audio)
        return audio, None, target

    def __len__(self) -> int:
        return len(self.data)