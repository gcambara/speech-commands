from einops import rearrange, repeat
import numpy as np
import torch
from torch import nn
from perceiver_pytorch import Perceiver
from transformers import Wav2Vec2ForPreTraining
from .layers import PostNormTransformer

class KWT(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, cfg, num_labels, time_size, freq_size):
        super().__init__()
        img_x, img_y = time_size, freq_size

        assert img_x % cfg.kwt_patch_x == 0, 'Image dimensions must be divisible by the patch size.'
        assert img_y % cfg.kwt_patch_y == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_x // cfg.kwt_patch_x) * (img_y // cfg.kwt_patch_y)
        patch_dim = cfg.kwt_channels * cfg.kwt_patch_x * cfg.kwt_patch_y

        assert cfg.kwt_pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.transformer = PostNormTransformer(dim=cfg.kwt_dim,
                                               depth=cfg.kwt_depth,
                                               heads=cfg.kwt_heads,
                                               dim_head=cfg.kwt_dim_head,
                                               mlp_dim=cfg.kwt_mlp_dim,
                                               dropout=cfg.kwt_dropout)

        self.lin_proj = nn.Linear(img_y, cfg.kwt_dim)

        self.g_feature = nn.Parameter(torch.randn(1, 1, cfg.kwt_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.kwt_dim))

        self.pool = cfg.kwt_pool
        self.mlp_head = nn.Linear(cfg.kwt_dim, num_labels)

    def forward(self, img):
        x = self.lin_proj(img)
        b, T, _ = x.shape

        g_features = repeat(self.g_feature, '() n d -> b n d', b = b)
        x = torch.cat((g_features, x), dim=1)
        x = x + self.pos_embedding

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

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

        if cfg.prc_freeze_latents:
            self.perceiver.latents.requires_grad = False

    def forward(self, x):
        return self.perceiver(x)

class PerceiverWav2Vec2(nn.Module):
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

        wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(cfg.teacher)
        wav2vec2_codevector = wav2vec2.quantizer.codevectors.squeeze()

        self.perceiver.latents = nn.Parameter(wav2vec2_codevector)

        if cfg.latent_weight_norm != 'none':
            self.perceiver.latents = self.normalize_weights(self.perceiver.latents, cfg.latent_weight_norm)

        if cfg.prc_freeze_latents:
            self.perceiver.latents.requires_grad = False

    def forward(self, x):
        return self.perceiver(x)

    def normalize_weights(self, weights, norm_type):
        if norm_type == 'kaiming': # min-max normalization, so values are between (-sqrt(1/k), sqrt(1/k))
            latent_dim = weights.size(-1)

            a = np.sqrt(1 / latent_dim)
            b = -a
            weights = a + (((weights - weights.min())*(b - a))/(weights.max() - weights.min()))
        return nn.Parameter(weights)
