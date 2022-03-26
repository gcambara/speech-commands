from einops import rearrange
import numpy as np
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram, MFCC

class Featurizer(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, cfg):
        super().__init__()
        # Feature extraction
        self.n_fft = cfg.n_fft
        self.n_mels = cfg.n_mels
        self.n_mfcc = cfg.n_mfcc
        self.deltas = cfg.deltas
        self.sr = cfg.sampling_rate
        self.win_length = int(self.sr * cfg.win_length)
        self.hop_length = int(self.sr * cfg.hop_length)
        self.chop_size = cfg.chop_size
        self.featurizer_post_norm = cfg.featurizer_post_norm

        self.featurizer = self.get_featurizer(cfg, cfg.featurizer)

    def get_featurizer(self, cfg, featurizer_names):
        featurizer_names = featurizer_names.split(',')
        if len(featurizer_names) == 1:
            featurizer_name = featurizer_names[0]
            if featurizer_name == 'mfsc':
                featurizer = MelFilterbank(sampling_rate=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, win_length=self.win_length, hop_length=self.hop_length, window_fn=torch.hamming_window, apply_log=False, norm=self.featurizer_post_norm)
            elif featurizer_name == 'log-mfsc':
                featurizer = MelFilterbank(sampling_rate=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, win_length=self.win_length, hop_length=self.hop_length, window_fn=torch.hamming_window, apply_log=True, norm=self.featurizer_post_norm)
            elif featurizer_name == 'mfcc':
                featurizer = Mfcc(sampling_rate=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, n_mfcc=self.n_mfcc, win_length=self.win_length, hop_length=self.hop_length, window_fn=torch.hamming_window, apply_log=False, norm=self.featurizer_post_norm, deltas=self.deltas)
            elif featurizer_name == 'log-mfcc':
                featurizer = Mfcc(sampling_rate=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, n_mfcc=self.n_mfcc, win_length=self.win_length, hop_length=self.hop_length, window_fn=torch.hamming_window, apply_log=True, norm=self.featurizer_post_norm, deltas=self.deltas)
            elif featurizer_name == 'waveform':
                featurizer = None
            elif featurizer_name == 'wavechops':
                featurizer = WaveChops(chop_size=self.chop_size)
            else:
                raise NotImplementedError
        else:
            featurizers = []
            for featurizer_name in featurizer_names:
                featurizers.append(self.get_featurizer(cfg, featurizer_name))
            featurizer = nn.ModuleList(featurizers)
        return featurizer

    def forward(self, x):
        if isinstance(self.featurizer, nn.ModuleList):
            out_feats = []
            for featurizer in self.featurizer:
                out_feat = featurizer(x)
                out_feats.append(out_feat)
            x = torch.cat(out_feats, dim=-1)
        else:
            x = self.featurizer(x)

        return x

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
    '''Input  = (B, T, C) '''
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

class WaveChops(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self,
                 chop_size=160
                 ):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x):
        b, t, c = x.shape
        x = x.view(b, int(t / self.chop_size), self.chop_size)
        return x