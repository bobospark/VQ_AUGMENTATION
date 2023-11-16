'''
VQ-Augmentation의 Encoder 코드
'''

import torch.nn as nn
from helper import ResidualBlock
import numpy as np
import torch


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        torch.manual_seed(args.seed)

        channels = [1024, 512, 256, 128]  

        layers = []
        layers.append(ResidualBlock(args.text_shape, 1024, normalize = False))
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            layers.append(ResidualBlock(in_channels, out_channels, normalize = True))



        layers.append(nn.Linear(128, 128))        

        self.model = nn.Sequential(*layers) #.to(device = args.device)
                
                
    def forward(self, real_x):
        text_hallucinated = self.model(real_x) 
        
        return text_hallucinated