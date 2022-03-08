from einops import rearrange
import numpy as np
import torch
from torch import nn

class LeNet(nn.Module):
    '''Input  = (B, T, C) '''
    '''Output = (B, T, C) '''
    def __init__(self, num_labels, time_size, freq_size):
        super().__init__()

        self.conv_blocks = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5),
                                        nn.MaxPool2d(2),
                                        nn.ReLU(),
                                        nn.Conv2d(20, 20, kernel_size=5),
                                        nn.MaxPool2d(2),
                                        nn.ReLU(),
                                        nn.Dropout2d(),
                                    )

        with torch.no_grad():
            _, _, time_size, freq_size = self.conv_blocks(torch.randn(1, 1, time_size, freq_size)).shape

        self.mlp = nn.Sequential(
            nn.Linear(20 * time_size * freq_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        x = rearrange(x, 'b t c -> b 1 t c')
        x = self.conv_blocks(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.mlp(x)
        return x