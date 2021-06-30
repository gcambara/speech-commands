'''Base class for the Speech Commands Detection models.'''

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from ..modules.features import MelFilterbank

class LightningModel(pl.LightningModule):
    '''Base class for the Speech Commands Detection models'''
    subclasses = {}

    def __init__(self):
        super(LightningModel, self).__init__()

    @classmethod
    def register_model(cls, model_name):
        def decorator(subclass):
            cls.subclasses[model_name] = subclass
            return subclass

        return decorator

    @classmethod
    def build_model(cls, cfg, num_classes):
        if cfg.arch not in cls.subclasses:
            raise ValueError(f"Unknown architecture name set in --arch: {cfg.arch}")
        return cls.subclasses[cfg.arch](cfg, num_classes)

@LightningModel.register_model('lenet')
class LeNet(LightningModel):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.num_classes = num_classes
        self.chunk_size = cfg.chunk_size
        self.lr = cfg.lr
        self.lr_scheduler = cfg.lr_scheduler
        self.lr_gamma = cfg.lr_gamma
        self.lr_step_size = cfg.lr_step_size
        self.n_fft = cfg.n_fft
        self.n_mels = cfg.n_mels
        self.sr = cfg.sampling_rate
        self.win_length = int(self.sr * cfg.win_length)
        self.hop_length = int(self.sr * cfg.hop_length)
        self.freq_size, self.time_size = self.compute_features_size()

        self.log_model_params = cfg.log_model_params

        self.mels = nn.Sequential(
            MelFilterbank(sampling_rate=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, win_length=self.win_length, hop_length=self.hop_length, window_fn=torch.hamming_window),
            nn.InstanceNorm2d(1)
        )

        self.features = nn.Sequential( #bs, 1, 40, 101
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(),
        )
  
        self.classifier = nn.Sequential(
            nn.Linear(20 * self.freq_size * self.time_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.mels(x)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)

        loss = self.loss(preds, targets)
        return {'loss': loss, 'preds': preds, 'targets': targets}

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)

        loss = self.loss(preds, targets)
        return {'loss': loss, 'preds': preds, 'targets': targets}

    def validation_epoch_end(self, outputs):
        self.log_avg_loss(outputs)
        self.log_accuracy(outputs)

    def log_accuracy(self, outputs):
        n_correct = 0
        n_samples_total = 0
        for index, batch_output in enumerate(outputs):
            preds = batch_output['preds']
            targets = batch_output['targets']
            pred_labs = preds.max(1).indices
            n_samples_total += targets.size(0)

            n_correct += torch.sum(pred_labs == targets)

        accuracy = n_correct / n_samples_total
        self.log(f'dev_accuracy', accuracy, on_epoch=True, prog_bar=True, sync_dist=True)

    def log_avg_loss(self, outputs):
        avg_loss = 0
        for index, batch_output in enumerate(outputs):
            batch_total_loss = batch_output['loss']
            avg_loss += batch_total_loss / len(outputs)

        self.log(f'dev_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def compute_features_size(self):
        freq_size = int(((self.n_mels - 5 + 1) / 2 - 5 + 1) / 2)

        timesteps = int((self.chunk_size - (self.n_fft // 2 + 1) + 2*self.hop_length) / self.hop_length) + 1
        time_size = int(((timesteps - 5 + 1) / 2 - 5 + 1) / 2)
        return freq_size, time_size