from einops import rearrange
import torch
from torch import nn
from audiomentations import Compose, AddBackgroundNoise, Resample, Shift

class WaveformAugmentations(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, cfg, background_noises_path=None):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        self.sampling_rate = cfg.sampling_rate
        self.augmentations = self.build_augmentations(cfg, background_noises_path)

    def forward(self, x):
        if self.augmentations:
            x = rearrange(x, '1 t -> t')
            x = x.numpy()
            x = torch.Tensor(self.augmentations(samples=x, sample_rate=self.sampling_rate))
            x = rearrange(x, 't -> 1 t')
        return x

    def build_augmentations(self, cfg, background_noises_path=None):
        augmentations = []

        if (cfg.time_shift_p > 0.0) and (cfg.time_shift_range > 0.0):
            fraction = cfg.time_shift_range / (self.chunk_size / self.sampling_rate)
            augmentations.append(Shift(min_fraction=-fraction,
                                       max_fraction=fraction,
                                       p=cfg.time_shift_p))

        if (cfg.resample_p > 0.0):
            min_sample_rate = int(self.sampling_rate * cfg.resample_min)
            max_sample_rate = int(self.sampling_rate * cfg.resample_max)
            augmentations.append(Resample(min_sample_rate=min_sample_rate,
                                          max_sample_rate=max_sample_rate,
                                          p=cfg.resample_p))

        if (cfg.background_noise_p > 0.0) and background_noises_path:
            augmentations.append(AddBackgroundNoise(sounds_path=background_noises_path,
                                                    min_snr_in_db=cfg.background_snr_min,
                                                    max_snr_in_db=cfg.background_snr_max,
                                                    noise_rms='relative',
                                                    p=cfg.background_noise_p))

        if augmentations != []:
            augmentations = Compose(augmentations)
        else:
            augmentations = None

        return augmentations

class PadTrim(object):
    def __init__(self, max_len, fill_value=0, channels_first=True):
        super(PadTrim, self).__init__()
        self.max_len = max_len
        self.fill_value = fill_value
        self.len_dim, self.ch_dim = int(channels_first), int(not channels_first)

    def __call__(self, tensor):
        if self.max_len > tensor.size(self.len_dim):
            padding = [self.max_len - tensor.size(self.len_dim)
                       if (i % 2 == 1) and (i // 2 != self.len_dim)
                       else 0
                       for i in range(4)]
            with torch.no_grad():
                tensor = torch.nn.functional.pad(tensor, padding, "constant", self.fill_value)
        elif self.max_len <= tensor.size(self.len_dim):
            tensor = tensor.narrow(self.len_dim, 0, self.max_len)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(max_len={0})'.format(self.max_len)