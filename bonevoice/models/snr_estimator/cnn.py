import torch
import torch.nn as nn
import torch.nn.functional as F

class StatisticsPooling(nn.Module):
    def __init__(self):
        super(StatisticsPooling, self).__init__()

    def forward(self, x):
        # x shape: (batch, channels, time)
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        # Concatenate mean and std along the channel dimension
        return torch.cat([mean, std], dim=1)

class SpeechModel(nn.Module):
    def __init__(self, latent_dim=128, n_inp=256):
        super(SpeechModel, self).__init__()
        self.latent_dim = latent_dim
        self.n_inp = n_inp

        # Encoder: Sequential Conv1d layers with ReLU
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=latent_dim,
                kernel_size=4,
                stride=1,
                padding=0  # 'valid' padding means no padding
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=4,
                stride=2,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=4,
                stride=2,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=4,
                stride=2,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=4,
                stride=2,
                padding=0
            ),
            nn.ReLU()
        )

        # Statistics Pooling
        self.stat_pooling = StatisticsPooling()

        # Encoder output block
        self.encoder_out = nn.Sequential(
            nn.Linear(n_inp, n_inp),
            nn.ReLU(),
            nn.Linear(n_inp, 1),
            nn.Sigmoid()
        )

    def forward(self, x, vibration):
        x = torch.stack([x, vibration], dim=1)
        # x shape: (batch, channels=2, time)
        x = self.encoder(x)  # Apply convolutional layers
        x = self.stat_pooling(x)  # Apply statistics pooling
        x = self.encoder_out(x)  # Apply final linear layers and sigmoid
        return x