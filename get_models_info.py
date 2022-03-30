import copy
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.model_summary import summarize, get_human_readable_count, get_formatted_model_size
from src.config.arguments import parse_arguments
from src.datamodules.speech_commands import SpeechCommandsDataModule
from src.models.lightning_model import LightningModel
from tqdm import tqdm
import time
import torch
from torchsummary import summary
import yaml

def forward_measure(cfg, model, input):
    fwd_measures = []
    total_fwd = cfg.warmup_fwd + cfg.n_fwd
    for i in tqdm(range(total_fwd)):
        if i < cfg.warmup_fwd:
            y = model(input)
        else:
            start_time = time.time()
            y = model(input)
            elapsed_time = time.time() - start_time
            fwd_measures.append(elapsed_time)

    fwd_measures = np.array(fwd_measures)
    mean_fwd_time = fwd_measures.mean()
    std_fwd_time = fwd_measures.std()

    return mean_fwd_time, std_fwd_time

def get_model_size(model, backend='pytorch-lightning'):
    if backend == 'pytorch-lightning':
        model_summary = summarize(model, max_depth=1)
        summary_data = model_summary._get_summary_data()
        total_parameters = model_summary.total_parameters
        trainable_parameters = model_summary.trainable_parameters
        model_size = model_summary.model_size
    else:
        param_size = 0
        total_parameters = 0
        trainable_parameters = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            total_parameters += param.nelement()
            if param.requires_grad:
                trainable_parameters += param.nelement()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size = (param_size + buffer_size) / 1024**2

    model_size = get_formatted_model_size(model_size)
    trainable_parameters = get_human_readable_count(trainable_parameters)
    total_parameters = get_human_readable_count(total_parameters)

    return model_size, trainable_parameters, total_parameters

def get_model_info(cfg, model, name):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("*****************************************")
    print(f"{name}\n")

    print(f"Measuring {name} model size and parameters...")
    model_size, trainable_parameters, total_parameters = get_model_size(model)

    print(f"Measuring {name} forward timings...")
    mean_fwd_time, std_fwd_time = forward_measure(cfg, model, input=torch.randn(1, 16000, 1))

    print(f"Model size: {model_size} MB")
    print(f"# Trainable parameters: {trainable_parameters}")
    print(f"# Total parameters: {total_parameters}")
    print(f"Mean fwd time = {mean_fwd_time} (+/-) {std_fwd_time} s")

    print("*****************************************\n")

def test_kwt1(args):
    cfg = copy.deepcopy(args)
    cfg.featurizer = 'mfcc'
    cfg.n_mfcc = 40
    cfg.win_length = 0.025
    cfg.hop_length = 0.010

    cfg.classifier = 'kwt'
    cfg.kwt_depth = 12
    cfg.kwt_dim = 64
    cfg.kwt_heads = 1
    cfg.kwt_mlp_dim = 256
    cfg.kwt_dropout = 0.0
    cfg.kwt_dim_head = 64
    cfg.kwt_patch_x = 1
    cfg.kwt_patch_y = 40
    cfg.kwt_channels = 1
    cfg.kwt_pool = 'cls'

    model = LightningModel(cfg)

    get_model_info(cfg, model, name='kwt_1')

def test_kwt2(args):
    cfg = copy.deepcopy(args)
    cfg.featurizer = 'mfcc'
    cfg.n_mfcc = 40
    cfg.win_length = 0.025
    cfg.hop_length = 0.010

    cfg.classifier = 'kwt'
    cfg.kwt_depth = 12
    cfg.kwt_dim = 128
    cfg.kwt_heads = 2
    cfg.kwt_mlp_dim = 512
    cfg.kwt_dropout = 0.0
    cfg.kwt_dim_head = 64
    cfg.kwt_patch_x = 1
    cfg.kwt_patch_y = 40
    cfg.kwt_channels = 1
    cfg.kwt_pool = 'cls'

    model = LightningModel(cfg)

    get_model_info(cfg, model, name='kwt_2')

def test_kwt3(args):
    cfg = copy.deepcopy(args)
    cfg.featurizer = 'mfcc'
    cfg.n_mfcc = 40
    cfg.win_length = 0.025
    cfg.hop_length = 0.010

    cfg.classifier = 'kwt'
    cfg.kwt_depth = 12
    cfg.kwt_dim = 192
    cfg.kwt_heads = 3
    cfg.kwt_mlp_dim = 768
    cfg.kwt_dropout = 0.0
    cfg.kwt_dim_head = 64
    cfg.kwt_patch_x = 1
    cfg.kwt_patch_y = 40
    cfg.kwt_channels = 1
    cfg.kwt_pool = 'cls'

    model = LightningModel(cfg)

    get_model_info(cfg, model, name='kwt_3')

def test_perceiver_base(args):
    cfg = copy.deepcopy(args)
    cfg.featurizer = 'log-mfsc'
    cfg.n_mels = 64
    cfg.win_length = 0.025
    cfg.hop_length = 0.010

    cfg.classifier = 'perceiver'
    cfg.prc_input_channels = cfg.n_mels
    cfg.prc_input_axis = 1
    cfg.prc_num_freq_bands = 12
    cfg.prc_max_freq = 10.
    cfg.prc_depth = 6
    cfg.prc_num_latents = 128
    cfg.prc_latent_dim = 256
    cfg.prc_cross_heads = 1
    cfg.prc_latent_heads = 8
    cfg.prc_cross_dim_head = 64
    cfg.prc_latent_dim_head = 64
    cfg.prc_attn_dropout = 0.0
    cfg.prc_ff_dropout = 0.0
    cfg.prc_weight_tie_layers = 1
    cfg.prc_fourier_encode_data = 1
    cfg.prc_self_per_cross_attn = 1

    model = LightningModel(cfg)

    get_model_info(cfg, model, name='perceiver_base')

def main():
    cfg = parse_arguments(stage='train')
    print(cfg)

    #os.makedirs(cfg.run_dir, exist_ok=True)

    pl.seed_everything(cfg.seed)
    cfg.use_cuda = cfg.use_cuda and torch.cuda.is_available()
    available_gpus = torch.cuda.device_count()
    if cfg.num_gpus > available_gpus:
        cfg.num_gpus = available_gpus
    if cfg.use_cuda:
        print(f"Using CUDA with {cfg.num_gpus} GPUs")
    else:
        print(f"Using CPU with {cfg.num_workers} workers")
        if cfg.precision != 32:
            print(f"Warning! Requested tensor precision is {cfg.precision}, but CPU only supports precision 32! Precision set to 32.")
            cfg.precision = 32
    if cfg.accelerator == 'ddp':
        plugins = DDPPlugin(find_unused_parameters=cfg.ddp_find_unused_parameters)
    else:
        plugins = None

    cfg.class_weights = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_perceiver_base(cfg)
    test_kwt1(cfg)
    test_kwt2(cfg)
    test_kwt3(cfg)

    # #summary(model.classifier.to(device), (1, 101, 64)) # the forward method of the model is supposed to get shape 4

if __name__ == '__main__':
    main()