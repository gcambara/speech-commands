import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.root = cfg.data
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.url = 'speech_commands_v0.02'
        self.chunk_size = cfg.chunk_size
        self.num_labels = cfg.num_labels

        self.batch_size = cfg.batch_size
        self.batch_size_dev = cfg.batch_size_dev
        self.batch_size_test = cfg.batch_size_test

        self.num_workers = cfg.num_workers
        self.shuffle = cfg.shuffle
        self.shuffle_dev = cfg.shuffle_dev
        self.shuffle_test = cfg.shuffle_test
        self.use_cuda = cfg.use_cuda

    def prepare_data(self):
        SPEECHCOMMANDS(self.root, url=self.url, download=True)

    def setup(self, stage=None):
        self.train_dataset = SPEECHCOMMANDS(self.root, url=self.url, download=False, subset='training')
        self.dev_dataset = SPEECHCOMMANDS(self.root, url=self.url, download=False, subset='validation')
        self.test_dataset = SPEECHCOMMANDS(self.root, url=self.url, download=False, subset='testing')

        if self.num_labels == 35:
            self.labels = sorted(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'])
        elif self.num_labels == 20:
            self.labels = sorted(['unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])
        elif self.num_labels == 10:
            self.labels = sorted(['unknown', 'left', 'right', 'yes', 'no', 'up', 'down', 'on', 'off', 'stop', 'go'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collater, pin_memory=self.use_cuda)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size_dev, drop_last=True, shuffle=self.shuffle_dev, num_workers=self.num_workers, collate_fn=self.collater, pin_memory=self.use_cuda)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, drop_last=False, shuffle=self.shuffle_test, num_workers=self.num_workers, collate_fn=self.collater, pin_memory=self.use_cuda)

    def collater(self, samples):
        waveforms = []
        labels = []
        for (waveform, _, label, *_) in samples:
            waveforms.append(waveform.squeeze())
            if label not in self.labels:
                label = 'unknown'
            labels.append(self.label_to_index(label))

        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True).unsqueeze(1)
        labels = torch.stack(labels)

        return waveforms, labels

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))

    def index_to_label(self, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]