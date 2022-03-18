from einops import rearrange
import random
import torch
from torch import nn
from torchaudio.transforms import FrequencyMasking, TimeMasking
from audiomentations import Compose, AddBackgroundNoise, Resample, Shift

class WaveformAugmentations(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, cfg, background_noises_path=None):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        self.sampling_rate = cfg.sampling_rate
        self.pad_trim = PadTrim(max_len=cfg.chunk_size, fill_value=0.0, channels_first=True)
        self.pre_augmentations, self.post_augmentations = self.build_augmentations(cfg, background_noises_path)
        if cfg.num_labels == 10:
            self.only_noise_p = cfg.only_noise_p
            self.only_noise_augmentation = AddBackgroundNoise(sounds_path=background_noises_path,
                                                              min_absolute_rms_in_db=-15.0,
                                                              max_absolute_rms_in_db=-0.1,
                                                              noise_rms='absolute',
                                                              p=1.0)
        else:
            self.only_noise_p = 0.0

    def forward(self, x):
        its_silence_sample = False
        if (self.only_noise_p > 0.0) and (random.uniform(0, 1) < self.only_noise_p):
            x = torch.zeros(x.size(-1))
            x = x.numpy()
            x = torch.Tensor(self.only_noise_augmentation(samples=x, sample_rate=self.sampling_rate))
            x = rearrange(x, 't -> 1 t')
            its_silence_sample = True
        else:
            if self.pre_augmentations:
                x = rearrange(x, '1 t -> t')
                x = x.numpy()
                x = torch.Tensor(self.pre_augmentations(samples=x, sample_rate=self.sampling_rate))
                x = rearrange(x, 't -> 1 t')

                # Some augmentations like resample change the size of the waveform, refit it
                if x.size(-1) != self.chunk_size:
                    x = self.pad_trim(x)

            if self.post_augmentations:
                x = rearrange(x, '1 t -> t')
                x = x.numpy()
                x = torch.Tensor(self.post_augmentations(samples=x, sample_rate=self.sampling_rate))
                x = rearrange(x, 't -> 1 t')

        return x, its_silence_sample

    def build_augmentations(self, cfg, background_noises_path=None):
        pre_augmentations, post_augmentations = [], []

        if (cfg.time_shift_p > 0.0) and (cfg.time_shift_range > 0.0):
            fraction = cfg.time_shift_range / (self.chunk_size / self.sampling_rate)
            pre_augmentations.append(Shift(min_fraction=-fraction,
                                       max_fraction=fraction,
                                       p=cfg.time_shift_p))

        if (cfg.resample_p > 0.0):
            min_sample_rate = int(self.sampling_rate * cfg.resample_min)
            max_sample_rate = int(self.sampling_rate * cfg.resample_max)
            pre_augmentations.append(Resample(min_sample_rate=min_sample_rate,
                                          max_sample_rate=max_sample_rate,
                                          p=cfg.resample_p))

        if (cfg.background_noise_p > 0.0) and background_noises_path:
            post_augmentations.append(AddBackgroundNoise(sounds_path=background_noises_path,
                                                    min_snr_in_db=cfg.background_snr_min,
                                                    max_snr_in_db=cfg.background_snr_max,
                                                    noise_rms='relative',
                                                    p=cfg.background_noise_p))

        if pre_augmentations != []:
            pre_augmentations = Compose(pre_augmentations)
        else:
            pre_augmentations = None

        if post_augmentations != []:
            post_augmentations = Compose(post_augmentations)
        else:
            post_augmentations = None

        return pre_augmentations, post_augmentations

class SpectrogramAugmentations(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, cfg):
        super().__init__()
        self.augmentations = self.build_augmentations(cfg)
        self.specaugment_p = cfg.specaugment_p

    def build_augmentations(self, cfg):
        augmentations = []

        if (cfg.specaugment_p > 0.0):
            for i in range(cfg.time_masks):
                augmentations.append(TimeMasking(time_mask_param=cfg.time_mask_size))

            for i in range(cfg.freq_masks):
                augmentations.append(FrequencyMasking(freq_mask_param=cfg.freq_mask_size))

        if augmentations != []:
            augmentations = nn.Sequential(*augmentations)
        else:
            augmentations = None

        return augmentations

    def forward(self, x):
        if self.augmentations and (random.uniform(0, 1) < self.specaugment_p):
            x = rearrange(x, 'b t c -> b c t')
            x = self.augmentations(x)
            x = rearrange(x, 'b c t -> b t c')
        return x

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