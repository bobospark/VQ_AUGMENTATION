'''
VQ-Augmentatio의 Dicriminator 코드
'''

import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        
        torch.manual_seed(args.seed)

        # self.label_embedding = nn.Embedding(args.num_classes, args.num_classes)

        # Copied from cgan.py
        self.model = nn.Sequential(
            nn.Linear(args.text_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, embedded_text):
        # Concatenate label embedding and image to produce input
        d_in = embedded_text # torch.cat((embedded_text.view(embedded_text.size(0), -1)), -1)  # , self.label_embedding(labels)
        validity = self.model(d_in)
        return validity