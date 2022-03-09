from einops import rearrange
import numpy as np
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram, MFCC

class MelFilterbank(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, sampling_rate=16000,
                 n_fft=512,
                 n_mels=40,
                 win_length=400,
                 hop_length=160,
                 window_fn=torch.hamming_window,
                 apply_log=True,
                 norm='instancenorm2d'):
        super().__init__()
        self.n_mels = n_mels
        self.epsilon = 1e-10
        self.apply_log = apply_log
        self.norm = norm

        self.fbank = MelSpectrogram(sample_rate=sampling_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, window_fn=window_fn, center=True)
        if self.norm == 'instancenorm2d':
            self.post_norm = nn.InstanceNorm2d(1)
        else:
            self.post_norm = None

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x = self.fbank(x)
        if self.apply_log:
            x = torch.log(x.add(self.epsilon))
        if self.post_norm:
            if self.norm == 'instancenorm2d':
                x = self.post_norm(x)
        x = rearrange(x, 'b 1 c t -> b t c')
        return x

class Mfcc(nn.Module):
    '''Input  = (B, T, T) '''
    '''Output = (B, T, C) '''
    def __init__(self,
                 sampling_rate=16000,
                 n_fft=512,
                 n_mfcc=20,
                 n_mels=40,
                 win_length=400,
                 hop_length=160,
                 window_fn=torch.hamming_window,
                 apply_log=False,
                 norm='instancenorm2d',
                 deltas=0):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.epsilon = 1e-10
        self.apply_log = apply_log
        self.norm = norm
        self.deltas = deltas

        self.melkwargs = {'n_fft': n_fft, 'n_mels': n_mels, 'win_length': win_length, 'hop_length': hop_length, 'window_fn': window_fn, 'center': True}
        self.mfcc = MFCC(sample_rate=sampling_rate, n_mfcc=self.n_mfcc, melkwargs=self.melkwargs)

        if self.norm == 'instancenorm2d':
            self.post_norm = nn.InstanceNorm2d(1)
        else:
            self.post_norm = None

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x = self.mfcc(x)
        if self.apply_log:
            x = torch.log(x.add(self.epsilon))
        if self.deltas > 0:
            raise NotImplementedError
            #mfcc = append_deltas(mfcc, self.deltas)
        if self.post_norm:
            if self.norm == 'instancenorm2d':
                x = self.post_norm(x)
        x = rearrange(x, 'b 1 c t -> b t c')
        return x

def append_deltas(feat, deltas):
    '''Input  = (B, C, T) '''
    '''Output = (B, C, T) '''
    for i in range(deltas):
        if i == 0:
            delta = compute_deltas(feat)
        else:
            delta = compute_deltas(delta)
        feat = torch.cat([feat, delta], dim=1)
    return feat