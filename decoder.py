'''
VQ-Augmentation의 Decoder 코드
'''

import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, BatchNorm, Swish
import numpy as np
import torch

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        torch.manual_seed(args.seed)
        channels = [128, 256, 512, 1024]  # 원래는 [1024, 512, 256, 128]였음
        
        layers = []
        layers.append(ResidualBlock(128, 128, normalize = False))
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            layers.append(ResidualBlock(in_channels, out_channels))



        layers.append(nn.Linear(1024, 1024))        
        # layers.append(Swish())  # 원래는 Swish()였음
        # layers.append(nn.Embedding(channels[-1], args.latent_dim, 3, 1, 1))
        
        self.model = nn.Sequential(*layers)
                
                
    def forward(self, x):
        # print(x.size())
        text_hallucinated = self.model(x)
        # text_hallucinated = text_hallucinated.view(text_hallucinated.size(0), *self.args.text_shape)

        return text_hallucinated