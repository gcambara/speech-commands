from einops import rearrange
import numpy as np
import torch
from torch import nn
from perceiver_pytorch import Perceiver

class LeNet(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, num_labels, time_size, freq_size):
        super().__init__()

        self.conv_blocks = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5),
                                        nn.MaxPool2d(2),
                                        nn.ReLU(),
                                        nn.Conv2d(20, 20, kernel_size=5),
                                        nn.MaxPool2d(2),
                                        nn.ReLU(),
                                        nn.Dropout2d(),
                                    )

        with torch.no_grad():
            _, _, time_size, freq_size = self.conv_blocks(torch.randn(1, 1, time_size, freq_size)).shape

        self.mlp = nn.Sequential(
            nn.Linear(20 * time_size * freq_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        x = rearrange(x, 'b t c -> b 1 t c')
        x = self.conv_blocks(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.mlp(x)
        return x

class PerceiverModel(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, cfg, num_labels):
        super().__init__()
        if cfg.prc_weight_tie_layers:
            weight_tie_layers = True
        else:
            weight_tie_layers = False

        if cfg.prc_fourier_encode_data:
            fourier_encode_data = True
        else:
            fourier_encode_data = False

        self.perceiver = Perceiver(input_channels=cfg.prc_input_channels,
                                   input_axis=cfg.prc_input_axis,
                                   num_freq_bands=cfg.prc_num_freq_bands,
                                   max_freq=cfg.prc_max_freq,
                                   depth=cfg.prc_depth,
                                   num_latents=cfg.prc_num_latents,
                                   latent_dim=cfg.prc_latent_dim,
                                   cross_heads=cfg.prc_cross_heads,
                                   latent_heads=cfg.prc_latent_heads,
                                   cross_dim_head=cfg.prc_cross_dim_head,
                                   latent_dim_head=cfg.prc_latent_dim_head,
                                   num_classes=num_labels,
                                   attn_dropout=cfg.prc_attn_dropout,
                                   ff_dropout=cfg.prc_ff_dropout,
                                   weight_tie_layers=weight_tie_layers,
                                   fourier_encode_data=fourier_encode_data,
                                   self_per_cross_attn=cfg.prc_self_per_cross_attn)

    def forward(self, x):
        return self.perceiver(x)