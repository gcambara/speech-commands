import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from src.config.arguments import parse_arguments
from src.datamodules.speech_commands import SpeechCommandsDataModule
from src.models.lightning_model import LightningModel
from src.models.distil_model import DistilModel
import torch
import yaml

def save_cfg(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg_yaml_path = os.path.join(out_dir, 'cfg.yaml')
    with open(cfg_yaml_path, 'w') as out_cfg:
        yaml.dump(vars(cfg), out_cfg)

    cfg_pickle_path = os.path.join(out_dir, 'cfg.pt')
    torch.save(cfg, cfg_pickle_path)

def main():
    cfg = parse_arguments(stage='train')
    print(cfg)

    os.makedirs(cfg.run_dir, exist_ok=True)

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

    dm = SpeechCommandsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    cfg.class_weights = dm.class_weights

    if cfg.model == 'base':
        model = LightningModel(cfg)
    elif cfg.model == 'distil':
        model = DistilModel(cfg)

    info = []
    for name, param in model.named_parameters():
        shape = list(param.shape)
        mean = param.mean().item()
        std = param.std().item()
        maximum = param.max().item()
        minimum = param.min().item()

        info.append([name, shape, mean, std, maximum, minimum])

    df = pd.DataFrame(info, columns =['name', 'shape', 'mean', 'std', 'max', 'min'])
    out_path = os.path.join(cfg.run_dir, f'{cfg.classifier}_weights_info.tsv')
    df.to_csv(out_path, sep='\t')


if __name__ == '__main__':
    main()