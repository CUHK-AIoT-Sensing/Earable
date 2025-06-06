from typing import List, Union
import torch
from torch import Tensor
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

class Model(nn.Module):
    def __init__(
            self, 
            inp_size: int=40, 
            num_layers: int=2, 
            hidden_size: int=200,
            is_causal: bool=False,
            ) -> None:
        """
        Args:
            inp_size (int): The shape of the input
            num_layers (int): The number of LSTM/biLSTM layers
            hidden_size (int): The hidden size of the LSTMs/biLSTMs 
            is_causal (bool): If True at time t the model will only look at 
            the frames from 0 till t-1 (LSTM), otherwise will be looking at 
            the previous and the future frames (biLSTM)
        """
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=inp_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional= not is_causal
        )
        self.output_layer = nn.Linear(
            in_features=hidden_size if is_causal else 2 * hidden_size,
            out_features=1
        )
        self.transform = MelSpectrogram(16000, n_fft=400, hop_length=200, win_length=400, n_mels=inp_size * 2, normalized=True)
    
    def forward(
            self, 
            x: Tensor,
            vibration: Tensor, 
            ) -> Tensor:
        """Performs forward pass for the input x.

        Args:
            x (Tensor): The input to the model of shape (B, M, D)
            lengths (Union[List[int], Tensor]): The lengths of 
            the input without padding.

        Returns:
            Tensor: The estimated SNR
        """
        # convert audio to spectrogram
        x = self.transform(x)
        vibration = self.transform(vibration)
        x = torch.cat([x, vibration], dim=-1)
        x = x.permute(0, 2, 1)  # (B, D, M) -> (B, M, D)
        output, (hn, cn) = self.lstm(x)
        output = output[:, -1, :]  # take the last time step
        output = self.output_layer(output)
        return output