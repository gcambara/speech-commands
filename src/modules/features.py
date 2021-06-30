import numpy as np
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

class MelFilterbank(nn.Module):
    '''Input  = (B, C, T) '''
    '''Output = (B, T, C) '''
    def __init__(self, sampling_rate=16000, n_fft=512, n_mels=40, win_length=400, hop_length=160, window_fn=torch.hamming_window):
        super().__init__()
        self.n_mels = n_mels
        self.epsilon = 1e-10

        self.fbank = MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, window_fn=window_fn, center=True)

    def forward(self, x):
        mel_fbank = self.fbank(x).squeeze(dim=1)
        mel_fbank = torch.log(mel_fbank.add(self.epsilon))
        mel_fbank = mel_fbank.unsqueeze(dim=1)
        return mel_fbank