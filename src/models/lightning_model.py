'''Base class for the Speech Commands Detection models.'''

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchmetrics
from ..modules.features import MelFilterbank
from ..modules.classifiers import LeNet
from vit_pytorch import ViT

class LightningModel(pl.LightningModule):
    '''Base class for the Speech Commands Detection models'''

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.num_labels = cfg.num_labels
        if self.num_labels == 20 or self.num_labels == 10:
            self.num_labels += 1 # sum 'unknown' class
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
        self.featurizer_post_norm = cfg.featurizer_post_norm

        self.log_model_params = cfg.log_model_params

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

        self.featurizer = self.get_featurizer(cfg.featurizer)
        _, time_size, freq_size = self.featurizer(torch.randn(1, self.chunk_size, 1)).shape

        self.classifier = self.get_classifier(cfg.classifier, time_size, freq_size)

        self.loss = self.get_loss(cfg.loss)

    def forward(self, x):
        if self.featurizer:
            x = self.featurizer(x)
        x = self.classifier(x)
        return x

    def _shared_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        loss = self.loss(preds, targets)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.train_acc(preds, targets)
        self.log(f'train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train_acc', self.train_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'preds': preds, 'targets': targets.detach()}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.valid_acc(preds, targets)
        self.log('dev_acc', self.valid_acc, on_step=False, on_epoch=True)
        return {'loss': loss, 'preds': preds, 'targets': targets.detach()}

    def validation_epoch_end(self, outputs):
        self.log_avg_loss(outputs)

    def get_featurizer(self, featurizer_name):
        if featurizer_name == 'mfsc':
            featurizer = MelFilterbank(sampling_rate=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, win_length=self.win_length, hop_length=self.hop_length, window_fn=torch.hamming_window, apply_log=False, norm=self.featurizer_post_norm)
        elif featurizer_name == 'log-mfsc':
            featurizer = MelFilterbank(sampling_rate=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, win_length=self.win_length, hop_length=self.hop_length, window_fn=torch.hamming_window, apply_log=True, norm=self.featurizer_post_norm)
        elif featurizer_name == 'waveform':
            featurizer = None
        else:
            raise NotImplementedError

        return featurizer

    def get_classifier(self, classifier_name, time_size, freq_size):
        if classifier_name == 'lenet':
            classifier = LeNet(self.num_labels, time_size, freq_size)
        else:
            raise NotImplementedError

        return classifier

    def get_loss(self, loss_name):
        if loss_name == 'cross-entropy':
            loss = nn.CrossEntropyLoss()
        return loss

    def log_avg_loss(self, outputs):
        avg_loss = 0
        for index, batch_output in enumerate(outputs):
            batch_total_loss = batch_output['loss']
            avg_loss += batch_total_loss / len(outputs)

        self.log(f'dev_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_scheduler == 'constant':
            return optimizer
        elif self.lr_scheduler == 'step_lr':
            scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
            return [optimizer], [scheduler]
        else:
            print(f"Warning! Unrecognized learning rate scheduler {self.lr_scheduler}. Training will be done without a scheduler.")
            return optimizer
