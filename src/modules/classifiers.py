from collections import Counter
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
import torch.nn.functional as F
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
    def __init__(self, cfg, num_labels, feat_size):
        super().__init__()
        if cfg.prc_weight_tie_layers:
            weight_tie_layers = True
        else:
            weight_tie_layers = False

        if cfg.prc_fourier_encode_data:
            fourier_encode_data = True
        else:
            fourier_encode_data = False

        if feat_size != int(cfg.prc_input_channels):
            self.feat_proj = nn.Linear(feat_size, int(cfg.prc_input_channels))
        else:
            self.feat_proj = None

        self.perceiver = Perceiver(input_channels=int(cfg.prc_input_channels),
                                   input_axis=int(cfg.prc_input_axis),
                                   num_freq_bands=int(cfg.prc_num_freq_bands),
                                   max_freq=float(cfg.prc_max_freq),
                                   depth=int(cfg.prc_depth),
                                   num_latents=int(cfg.prc_num_latents),
                                   latent_dim=int(cfg.prc_latent_dim),
                                   cross_heads=int(cfg.prc_cross_heads),
                                   latent_heads=int(cfg.prc_latent_heads),
                                   cross_dim_head=int(cfg.prc_cross_dim_head),
                                   latent_dim_head=int(cfg.prc_latent_dim_head),
                                   num_classes=num_labels,
                                   attn_dropout=float(cfg.prc_attn_dropout),
                                   ff_dropout=float(cfg.prc_ff_dropout),
                                   weight_tie_layers=weight_tie_layers,
                                   fourier_encode_data=fourier_encode_data,
                                   self_per_cross_attn=int(cfg.prc_self_per_cross_attn))

        if cfg.prc_freeze_latents:
            self.perceiver.latents.requires_grad = False

    def forward(self, x):
        if self.feat_proj:
            x = self.feat_proj(x)
        return self.perceiver(x)

class PerceiverWav2Vec2(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, cfg, num_labels, feat_size):
        super().__init__()
        if cfg.prc_weight_tie_layers:
            weight_tie_layers = True
        else:
            weight_tie_layers = False

        if cfg.prc_fourier_encode_data:
            fourier_encode_data = True
        else:
            fourier_encode_data = False

        if feat_size != int(cfg.prc_input_channels):
            self.feat_proj = nn.Linear(feat_size, int(cfg.prc_input_channels))
        else:
            self.feat_proj = None

        self.perceiver = Perceiver(input_channels=int(cfg.prc_input_channels),
                                   input_axis=int(cfg.prc_input_axis),
                                   num_freq_bands=int(cfg.prc_num_freq_bands),
                                   max_freq=float(cfg.prc_max_freq),
                                   depth=int(cfg.prc_depth),
                                   num_latents=int(cfg.prc_num_latents),
                                   latent_dim=int(cfg.prc_latent_dim),
                                   cross_heads=int(cfg.prc_cross_heads),
                                   latent_heads=int(cfg.prc_latent_heads),
                                   cross_dim_head=int(cfg.prc_cross_dim_head),
                                   latent_dim_head=int(cfg.prc_latent_dim_head),
                                   num_classes=num_labels,
                                   attn_dropout=float(cfg.prc_attn_dropout),
                                   ff_dropout=float(cfg.prc_ff_dropout),
                                   weight_tie_layers=weight_tie_layers,
                                   fourier_encode_data=fourier_encode_data,
                                   self_per_cross_attn=int(cfg.prc_self_per_cross_attn))

        wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(cfg.teacher)
        wav2vec2_codevector = wav2vec2.quantizer.codevectors.squeeze()

        if cfg.clusterize_latents:
            wav2vec2_codevector = self.clusterize_latents(wav2vec2_codevector,
                                                          int(cfg.prc_num_latents))

        self.perceiver.latents = nn.Parameter(wav2vec2_codevector)

        if cfg.latent_weight_norm != 'none':
            self.perceiver.latents = self.normalize_weights(self.perceiver.latents, cfg.latent_weight_norm)

        if cfg.prc_freeze_latents:
            self.perceiver.latents.requires_grad = False

        if cfg.use_w2v2_latents:
           self.latent_extractor = wav2vec2.wav2vec2.feature_extractor
           self.latent_extractor._freeze_parameters()
        else:
           self.latent_extractor = None

    def forward(self, x):
        if self.latent_extractor:
            x = rearrange(x, 'b t 1 -> b t')
            x = self.latent_extractor(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.feat_proj:
            x = self.feat_proj(x)
        return self.perceiver(x)

    def normalize_weights(self, weights, norm_type):
        if norm_type == 'kaiming': # min-max normalization, so values are between (-sqrt(1/k), sqrt(1/k))
            latent_dim = weights.size(-1)

            a = np.sqrt(1 / latent_dim)
            b = -a
            weights = a + (((weights - weights.min())*(b - a))/(weights.max() - weights.min()))
        return nn.Parameter(weights)

    def process_latents(self, latents, num_latents, latent_dim, mode='none'):
        src_num, src_dim = latents.shape

        if src_num == num_latents and src_dim == latent_dim:
            pass
        else:
            num_factor = src_num // num_latents
            dim_factor = src_dim // latent_dim

            if mode == 'avg_pool':
                if num_factor > 1:
                    latents = rearrange(latents, 'n d -> 1 d n')
                    latents = F.avg_pool1d(latents,
                                           kernel_size=num_factor,
                                           stride=num_factor)
                    latents = rearrange(latents, '1 d n -> n d')
                if dim_factor > 1:
                    latents = rearrange(latents, 'n d -> 1 n d')
                    latents = F.avg_pool1d(latents,
                                           kernel_size=dim_factor,
                                           stride=dim_factor)
                    latents = rearrange(latents, '1 n d -> n d')
            elif mode == 'random_sample':
                if src_num != num_latents:
                    perm = torch.randperm(src_num)
                    idx = perm[:num_latents]
                    latents = latents[idx]
                if src_dim != latent_dim:
                    perm = torch.randperm(src_dim)
                    idx = perm[:latent_dim]
                    latents = latents[idx]
            elif mode == 'pile_up':
                if num_factor > 1:
                    latents = latents.view(num_latents, src_dim * num_factor)

        return latents

    def clusterize_latents(self, latents, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(latents.detach().numpy())
        label_counts = Counter(kmeans.labels_)
        new_latents = torch.zeros(n_clusters, latents.size(-1))

        for index, label in enumerate(kmeans.labels_):
            count = label_counts[label]
            new_latents[label, :] += latents[index, :] / count

        return new_latents

class MultiPerceiverWav2Vec2(nn.Module):
    ''' Multi block Perceiver. At least set the number of layers with --prc_depth 2,2,2.
        If other arguments are not specified, they will be repeated along the number of total blocks.'''
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, cfg, num_labels, feat_size):
        super().__init__()
        self.latent_weight_norm = cfg.latent_weight_norm
        self.prc_freeze_latents = cfg.prc_freeze_latents
        self.multi_perceiver, latent_dim = self.build_multi_perceiver(cfg, num_labels)

        if feat_size != int(cfg.prc_input_channels):
            self.feat_proj = nn.Linear(feat_size, int(cfg.prc_input_channels))
        else:
            self.feat_proj = None

        if cfg.teacher != 'none':
            wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(cfg.teacher)
            wav2vec2_codevector = wav2vec2.quantizer.codevectors.squeeze()
            self.process_latents(wav2vec2_codevector,
                                 int(cfg.prc_num_latents),
                                 int(cfg.prc_latent_dim),
                                 cfg.latent_process_mode)

        self.to_logits = nn.Sequential(
                                        Reduce('b n d -> b d', 'mean'),
                                        nn.LayerNorm(latent_dim),
                                        nn.Linear(latent_dim, num_labels)
                                     )

    def forward(self, x):
        if self.feat_proj:
            x = self.feat_proj(x)

        latents = []
        for perce in self.multi_perceiver:
            latent = perce(x)
            latents.append(latent)

        x = torch.cat(latents, dim=1)
        return self.to_logits(x)

    def build_multi_perceiver(self, cfg, num_labels):
        multi_perceiver_args = [cfg.prc_input_channels,
                                cfg.prc_input_axis,
                                cfg.prc_num_freq_bands,
                                cfg.prc_max_freq,
                                cfg.prc_depth,
                                cfg.prc_num_latents,
                                cfg.prc_latent_dim,
                                cfg.prc_cross_heads,
                                cfg.prc_latent_heads,
                                cfg.prc_cross_dim_head,
                                cfg.prc_latent_dim_head,
                                num_labels,
                                cfg.prc_attn_dropout,
                                cfg.prc_ff_dropout,
                                cfg.prc_weight_tie_layers,
                                cfg.prc_fourier_encode_data,
                                cfg.prc_self_per_cross_attn]

        layers = str(cfg.prc_depth).split(',')
        n_blocks = len(layers)

        new_args = []
        for arg in multi_perceiver_args:
            arg_list = str(arg).split(',')
            if len(arg_list) == n_blocks:
                pass
            elif len(arg_list) == 1:
                arg_list = [arg] * n_blocks
            else:
                raise NotImplementedError(f"Error! Argument size {arg} does not match with the number of expected blocks {n_blocks}")

            new_args.append(arg_list)

        blocks = []
        for i in range(n_blocks):
            if int(new_args[14][i]):
                weight_tie_layers = True
            else:
                weight_tie_layers = False

            if int(new_args[15][i]):
                fourier_encode_data = True
            else:
                fourier_encode_data = False

            perceiver_block = Perceiver(input_channels=int(new_args[0][i]),
                                        input_axis=int(new_args[1][i]),
                                        num_freq_bands=int(new_args[2][i]),
                                        max_freq=float(new_args[3][i]),
                                        depth=int(new_args[4][i]),
                                        num_latents=int(new_args[5][i]),
                                        latent_dim=int(new_args[6][i]),
                                        cross_heads=int(new_args[7][i]),
                                        latent_heads=int(new_args[8][i]),
                                        cross_dim_head=int(new_args[9][i]),
                                        latent_dim_head=int(new_args[10][i]),
                                        num_classes=int(new_args[11][i]),
                                        attn_dropout=float(new_args[12][i]),
                                        ff_dropout=float(new_args[13][i]),
                                        weight_tie_layers=weight_tie_layers,
                                        fourier_encode_data=fourier_encode_data,
                                        self_per_cross_attn=int(new_args[16][i]),
                                        final_classifier_head=False)
            blocks.append(perceiver_block)
        multi_perceiver = nn.Sequential(*blocks)

        last_latent_dim = int(new_args[6][-1])
        return multi_perceiver, last_latent_dim

    def normalize_weights(self, weights, norm_type):
        if norm_type == 'kaiming': # min-max normalization, so values are between (-sqrt(1/k), sqrt(1/k))
            latent_dim = weights.size(-1)

            a = np.sqrt(1 / latent_dim)
            b = -a
            weights = a + (((weights - weights.min())*(b - a))/(weights.max() - weights.min()))
        return nn.Parameter(weights)

    def process_latents(self, latents, num_latents, latent_dim, mode='none'):
        src_num, src_dim = latents.shape

        # gcambara: for now, simply divide the latents by the number of blocks
        n_blocks = len(self.multi_perceiver)
        assert src_num / num_latents == n_blocks, f"Error! Latents from wav2vec2.0 divided by num_latents should be equal to the number of blocks. # w2v2 latents = {src_num} | # latents = {num_latents} | n_blocks = {n_blocks}"

        for i in range(n_blocks):
            sub_latents = latents[i*num_latents:(i + 1)*num_latents]
            self.multi_perceiver[i].latents = nn.Parameter(sub_latents)

            if self.latent_weight_norm != 'none':
                self.multi_perceiver[i].latents = self.normalize_weights(self.multi_perceiver[i].latents, self.latent_weight_norm)

            if self.prc_freeze_latents:
                self.multi_perceiver[i].latents.requires_grad = False
