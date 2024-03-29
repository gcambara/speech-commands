'''Base class for the Speech Commands Detection models.'''

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
import torch
from torch import nn
import torch.nn.functional as F
from torch_optimizer import Lamb
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torchmetrics
from ..datamodules.augmentations import SpectrogramAugmentations
from ..modules.features import Featurizer
from ..modules.classifiers import KWT, LeNet, MultiPerceiverWav2Vec2, PerceiverModel, PerceiverWav2Vec2
from ..optim.schedulers import ConsecutiveLR

class LightningModel(pl.LightningModule):
    '''Base class for the Speech Commands Detection models'''

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Input and output sizes
        self.num_labels = cfg.num_labels
        if self.num_labels == 20 or self.num_labels == 10:
            self.num_labels += 2 # sum 'unknown' and 'silence' class
        self.chunk_size = cfg.chunk_size

        # Optimizer options
        self.optimizer = cfg.optimizer
        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        self.weight_decay = cfg.weight_decay
        self.optimizer_eps = cfg.optimizer_eps
        self.lamb_clamp = cfg.lamb_clamp

        # LR scheduling
        self.lr = cfg.lr
        self.lr_scheduler = cfg.lr_scheduler
        self.lr_gamma = cfg.lr_gamma
        self.lr_milestones = [int(milestone) for milestone in cfg.lr_milestones.split(',')]
        self.schedulers = cfg.schedulers.split(',')
        self.lr_step_size = cfg.lr_step_size
        self.lr_min = cfg.lr_min
        self.lr_max_epochs = cfg.lr_max_epochs
        self.lr_warmup_epochs = cfg.lr_warmup_epochs
        self.max_epochs = cfg.max_epochs

        # Spectrogram augmentations
        if cfg.featurizer != 'waveform':
            self.spec_augments = SpectrogramAugmentations(cfg)
        else:
            self.spec_augments = None

        # Metrics
        self.log_model_params = cfg.log_model_params
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        # Build models and losses
        self.wav_normalization = self.get_wav_normalization(cfg.wav_norm)
        if cfg.featurizer != 'waveform':
            self.featurizer = Featurizer(cfg)
        else:
            self.featurizer = None
        if self.featurizer:
            with torch.no_grad():
                _, time_size, freq_size = self.featurizer(torch.randn(1, self.chunk_size, 1)).shape
        else:
            time_size, freq_size = self.chunk_size, 1

        self.classifier = self.get_classifier(cfg.classifier, time_size, freq_size)

        if hasattr(cfg, 'label_smoothing'):
            self.label_smoothing = cfg.label_smoothing
        else:
            self.label_smoothing = 0.0
        self.class_weights = cfg.class_weights
        self.loss = self.get_loss(cfg.loss)

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
        self.log('dev_acc', self.valid_acc, on_step=False, on_epoch=True)
        return {'loss': loss, 'preds': preds.detach(), 'targets': targets.detach()}

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.test_acc(preds, targets)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return {'loss': loss, 'preds': preds.detach(), 'targets': targets.detach()}

    def validation_epoch_end(self, outputs):
        self.log_avg_loss(outputs)
        self.log_confusion_matrix(outputs)

    def log_confusion_matrix(self, outputs):
        all_preds, all_targets = [], []
        for index, batch_output in enumerate(outputs):
            batch_preds = batch_output['preds']
            pred_ids = torch.argmax(batch_preds, dim=-1)
            target_ids = batch_output['targets']

            all_preds.append(pred_ids)
            all_targets.append(target_ids)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        get_confusion_matrix = torchmetrics.ConfusionMatrix(self.num_labels,
                                                            normalize='true').to(self.device)
        confusion_matrix = get_confusion_matrix(all_preds, all_targets)

        labels = self.trainer.datamodule.labels

        df_confusion_matrix = pd.DataFrame(confusion_matrix.cpu().numpy(),
                                           index=[i for i in labels],
                                           columns = [i for i in labels])

        plt.figure(figsize = (12,7))
        fig = sn.heatmap(df_confusion_matrix).get_figure()
        plt.tight_layout()
        self.logger[0].experiment.add_figure('dev_conf_matrix', fig, self.current_epoch)
        plt.clf()

    def get_classifier(self, classifier_name, time_size, freq_size):
        if classifier_name == 'lenet':
            classifier = LeNet(self.num_labels, time_size, freq_size)
        elif classifier_name == 'perceiver':
            classifier = PerceiverModel(self.cfg, self.num_labels, feat_size=freq_size)
        elif classifier_name == 'perceiver_w2v2':
            classifier = PerceiverWav2Vec2(self.cfg, self.num_labels, feat_size=freq_size)
        elif classifier_name == 'multi_perceiver_w2v2':
            classifier = MultiPerceiverWav2Vec2(self.cfg, self.num_labels, feat_size=freq_size)
        elif classifier_name == 'kwt':
            classifier = KWT(self.cfg, self.num_labels, time_size, freq_size)
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
        elif optimizer_name == 'lamb':
            optimizer = Lamb(self.parameters(),
                             lr=self.lr,
                             betas=(self.beta1, self.beta2),
                             eps=self.optimizer_eps,
                             weight_decay=self.weight_decay,
                             clamp_value=self.lamb_clamp)
        else:
            raise NotImplementedError

        return optimizer

    def get_scheduler(self, optimizer, scheduler_name):
        if scheduler_name == 'step_lr':
            scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        elif scheduler_name == 'cosine':
            if self.lr_warmup_epochs > 0:
                 scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                           warmup_epochs=self.lr_warmup_epochs,
                                                           max_epochs=self.lr_max_epochs,
                                                           eta_min=self.lr_min)
            else:
                scheduler = CosineAnnealingLR(optimizer,
                                              T_max=self.lr_max_epochs,
                                              eta_min=self.lr_min)
        elif scheduler_name == 'consecutive':
            schedulers = []
            for sub_scheduler in self.schedulers:
                schedulers.append(self.get_scheduler(optimizer, sub_scheduler))
            scheduler = ConsecutiveLR(optimizer, schedulers, milestones=self.lr_milestones)
        else:
            raise NotImplementedError(f"Warning! Unrecognized learning rate scheduler {scheduler_name}. Training will be done without a scheduler.")

        return scheduler

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
        else:
            scheduler = self.get_scheduler(optimizer, self.lr_scheduler)
            return [optimizer], [scheduler]
