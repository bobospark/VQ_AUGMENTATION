'''
VQ-Augmentation의 Encoder 코드
'''

import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, BatchNorm, Swish
import numpy as np
import torch


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        torch.manual_seed(args.seed)
        # self.label_emb = nn.Embedding(args.num_classes, args.num_classes).to(device = args.device)  # 여긷호 seed 안맞음

        channels = [1024, 512, 256, 128]  # 원래는 [128, 256, 512, 1024]였음

        layers = []
        layers.append(ResidualBlock(args.text_shape, 1024, normalize = False))
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            layers.append(ResidualBlock(in_channels, out_channels, normalize = True))



        layers.append(nn.Linear(128, 128))        
        # layers.append(nn.Tanh())  # 원래는 Swish()였음. 마지막을 Tanh로 하는 이유가 있을까?

        self.model = nn.Sequential(*layers) #.to(device = args.device)
                
                
    def forward(self, x):
        # print(x.size())
        # encoded_input = torch.cat((self.label_emb(labels), x), -1)  # 말썽투성이
        encoded_input = x  #.reshape(x.shape[0], -1)  # (batch_size, 1024, 1, 1) -> (batch_size, 1024)
        text_hallucinated = self.model(encoded_input)  # 여기까지 완료
        # text_hallucinated = text_hallucinated.view(text_hallucinated.shape)

        return text_hallucinated