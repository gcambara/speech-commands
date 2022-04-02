''' Distil model.'''

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torchmetrics
from ..datamodules.augmentations import SpectrogramAugmentations
from ..modules.encoders import Wav2Vec2
from ..modules.features import Featurizer
from ..modules.classifiers import LeNet, PerceiverModel
from .lightning_model import LightningModel

class DistilModel(LightningModel):
    '''Base class for the Speech Commands Detection models'''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sampling_rate = cfg.sampling_rate
        self.teacher = self.load_teacher(cfg.teacher, zoo=cfg.teacher_zoo)

        # wav2vec2.0 arguments
        self.w2v2_target_state = cfg.w2v2_target_state

    def load_teacher(self, teacher_url_path, zoo):
        if 'wav2vec2' in teacher_url_path:
            teacher = Wav2Vec2(url=teacher_url_path, zoo=zoo)
        else:
            raise NotImplementedError

        return teacher

    def forward(self, x):
        if self.wav_normalization:
            x = self.wav_normalization(x)
        if self.featurizer:
            x = self.featurizer(x)
        if self.spec_augments and self.training:
            x = self.spec_augments(x)
        x = self.classifier(x)
        return x

    def _shared_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        print(preds.shape)
        print(x.shape)
        with torch.no_grad():
            teacher_feats = self.teacher(x, target_state=self.w2v2_target_state)
        print(teacher_feats.shape)
        exit()
        loss = self.loss(preds, targets)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.train_acc(preds, targets)
        self.log(f'train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train_acc', self.train_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'preds': preds.detach(), 'targets': targets.detach()}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.valid_acc(preds, targets)
        self.log('dev_acc', self.valid_acc, on_step=False, on_epoch=True)
        return {'loss': loss, 'preds': preds.detach(), 'targets': targets.detach()}

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.test_acc(preds, targets)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return {'loss': loss, 'preds': preds.detach(), 'targets': targets.detach()}

    def validation_epoch_end(self, outputs):
        self.log_avg_loss(outputs)

    def get_classifier(self, classifier_name, time_size, freq_size):
        if classifier_name == 'lenet':
            classifier = LeNet(self.num_labels, time_size, freq_size)
        elif classifier_name == 'perceiver':
            classifier = PerceiverModel(self.cfg, self.num_labels)
        else:
            raise NotImplementedError

        return classifier

    def get_wav_normalization(self, norm_name):
        if norm_name == 'layernorm':
            norm = nn.LayerNorm(normalized_shape=[self.chunk_size, 1])
        elif norm_name == 'none':
            norm = None
        else:
            raise NotImplementedError

        return norm

    def get_loss(self, loss_name):
        if loss_name == 'cross-entropy':
            loss = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=self.label_smoothing)
        return loss

    def get_optimizer(self, optimizer_name):
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.optimizer_eps,
                                         weight_decay=self.weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(),
                                         lr=self.lr,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.optimizer_eps,
                                         weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        return optimizer

    def log_avg_loss(self, outputs):
        avg_loss = 0
        for index, batch_output in enumerate(outputs):
            batch_total_loss = batch_output['loss']
            avg_loss += batch_total_loss / len(outputs)

        self.log(f'dev_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = self.get_optimizer(self.optimizer)
        if self.lr_scheduler == 'constant':
            return optimizer
        elif self.lr_scheduler == 'step_lr':
            scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
            return [optimizer], [scheduler]
        elif self.lr_scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.lr_max_epochs, eta_min=self.lr_min)
            return [optimizer], [scheduler]
        else:
            print(f"Warning! Unrecognized learning rate scheduler {self.lr_scheduler}. Training will be done without a scheduler.")
            return optimizer
