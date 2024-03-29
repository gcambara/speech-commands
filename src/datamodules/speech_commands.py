from einops import rearrange
import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio.datasets import SPEECHCOMMANDS
from .augmentations import PadTrim, WaveformAugmentations
from tqdm import tqdm

class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, cfg, root, url, download, subset=None):
        super().__init__(root, url=url, download=download, subset=subset)
        self.subset = subset
        self.background_noises_path = os.path.join(root, 'SpeechCommands', url, '_background_noise_')
        self.augmentations = WaveformAugmentations(cfg, background_noises_path=self.background_noises_path)
        self.pad_trim = PadTrim(max_len=cfg.chunk_size, fill_value=0.0, channels_first=True)

    def __getitem__(self, index):
        audio, sample_rate, label, speaker_id, utterance_number = super().__getitem__(index)

        if self.augmentations and self.subset == 'training':
            audio, its_silence_sample = self.augmentations(audio)

            if its_silence_sample:
                label = 'silence'

        audio = self.pad_trim(audio)

        return audio, sample_rate, label, speaker_id, utterance_number

class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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

        self.class_weights = cfg.class_weights
        self.class_weights_batches = cfg.class_weights_batches

        self.weighted_sampler = cfg.weighted_sampler
        self.weighted_sampler_unk_weight = cfg.weighted_sampler_unk_weight

    def prepare_data(self):
        SpeechCommands(self.cfg, self.root, url=self.url, download=True)

    def setup(self, stage=None):
        self.train_dataset = SpeechCommands(self.cfg, self.root, url=self.url, download=False, subset='training')
        self.dev_dataset = SpeechCommands(self.cfg, self.root, url=self.url, download=False, subset='validation')
        self.test_dataset = SpeechCommands(self.cfg, self.root, url=self.url, download=False, subset='testing')

        if self.num_labels == 35:
            self.labels = sorted(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'])
        elif self.num_labels == 20:
            self.labels = sorted(['unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])
        elif self.num_labels == 10:
            self.labels = sorted(['unknown', 'left', 'right', 'yes', 'no', 'up', 'down', 'on', 'off', 'stop', 'go', 'silence'])

        if self.class_weights or self.weighted_sampler:
            self.class_weights, self.idx_sample = self.get_class_weights()
        else:
            self.class_weights, self.idx_sample = None, None

        if self.weighted_sampler:
            self.sampler = self.get_weighted_random_sampler()
            self.shuffle = None
        else:
            self.sampler = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, sampler=self.sampler, num_workers=self.num_workers, collate_fn=self.collater, pin_memory=self.use_cuda)

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
        waveforms = rearrange(waveforms, 'b c t -> b t c')
        labels = torch.stack(labels)

        return waveforms, labels

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))

    def index_to_label(self, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]

    def get_class_weights(self):
        idx_sample = np.zeros(len(self.train_dataset))
        if self.weighted_sampler:
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=None, num_workers=self.num_workers, collate_fn=self.collater, pin_memory=self.use_cuda)
            weights = torch.ones(len(self.labels))
            weights[self.label_to_index('unknown')] = self.weighted_sampler_unk_weight
            counter = 0
            for _, labels in tqdm(train_loader):
                for label in labels:
                    idx_sample[counter] = label
                    counter += 1
        else:
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=self.collater, pin_memory=self.use_cuda)
            weights = torch.zeros(len(self.labels))

            n_samples = 0
            print(f"Computing class weights for {self.class_weights_batches} batches...")
            counter = 0
            for i, (_, labels) in tqdm(enumerate(train_loader)):
                for label in labels:
                    weights[label] += 1
                    idx_sample[counter] = label
                    counter += 1
                n_samples += len(labels)

                if i == self.class_weights_batches:
                    break

            weights = n_samples / (len(self.labels) * weights)

        return weights, idx_sample

    def get_weighted_random_sampler(self):
        weights_sample = np.zeros(len(self.train_dataset))

        for i in range(0, len(self.labels)):
            weights_sample[np.where(self.idx_sample == i)] = self.class_weights[i]

        assert (len(weights_sample) == len(self.train_dataset))

        return WeightedRandomSampler(weights_sample, len(weights_sample), replacement=True)


