from einops import rearrange
import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2ForPreTraining

class Wav2Vec2(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, url, zoo):
        super().__init__()
        self.wav2vec2 = self.load_wav2vec2(url, zoo)

    def forward(self, x, target_state=-1):
        x = rearrange(x, 'b t 1 -> b t')
        
        if target_state == -1:
            output_hidden_states = False
        else:
            output_hidden_states = True
        x = self.wav2vec2(x, output_hidden_states=output_hidden_states)

        if target_state == -1:
            x = x['projected_states']
        else:
            x = x['hidden_states'][target_state]

        return x

    def load_wav2vec2(self, url, zoo):
        if zoo == 'huggingface':
            teacher = Wav2Vec2ForPreTraining.from_pretrained(url)
        else:
            raise NotImplementedError

        return teacher