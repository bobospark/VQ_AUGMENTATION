'''
VQ-Augmentation의 Decoder 코드
'''

import torch.nn as nn
from helper import ResidualBlock
import numpy as np
import torch

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        torch.manual_seed(args.seed)
        channels = [128, 256, 512, 1024]  
        
        layers = []
        layers.append(ResidualBlock(128, 128, normalize = False))
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            layers.append(ResidualBlock(in_channels, out_channels))

        layers.append(nn.Linear(1024, 1024))        
        
        self.model = nn.Sequential(*layers)
                
                
    def forward(self, encoded_input):
        text_hallucinated = self.model(encoded_input)

        return text_hallucinated