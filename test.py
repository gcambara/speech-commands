import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from src.config.arguments import parse_arguments
from src.datamodules.speech_commands import SpeechCommandsDataModule
from src.models.lightning_model import LightningModel
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
    cfg = parse_arguments(stage='test')

    model_cfg = torch.load(cfg.cfg)

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

    model_cfg.data = cfg.data
    dm = SpeechCommandsDataModule(model_cfg)
    dm.prepare_data()
    dm.setup()

    model = LightningModel(model_cfg)

    logger = [pl.loggers.TensorBoardLogger(save_dir=cfg.run_dir, name='tensorboard'),
              pl.loggers.CSVLogger(save_dir=cfg.run_dir, name='csv_logger')]

    save_cfg(cfg, logger[0].log_dir)

    trainer = pl.Trainer(accelerator=cfg.accelerator, default_root_dir=cfg.run_dir, gpus=cfg.num_gpus, deterministic=False, limit_train_batches=cfg.limit_train_batches, logger=logger, max_epochs=cfg.max_epochs, overfit_batches=cfg.overfit_batches, plugins=plugins, precision=cfg.precision, profiler=None, track_grad_norm=2)
    print("Testing on validation set...")
    trainer.test(model, ckpt_path=cfg.ckpt, test_dataloaders=dm.val_dataloader())
    print("Testing on test set...")
    trainer.test(model, ckpt_path=cfg.ckpt, test_dataloaders=dm.test_dataloader())

if __name__ == '__main__':
    main()