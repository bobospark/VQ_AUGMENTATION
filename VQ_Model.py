'''


모든 VQAugmentation에 들어가는 코드 통합시키는 곳
'''


import torch
# from transformers import DistilBertForSequenceClassification, 
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook

class VQAugmentation(nn.Module):
    def __init__(self, args):
        super(VQAugmentation, self).__init__()

        self.encoder = Encoder(args).to(device = args.device)
        self.decoder = Decoder(args).to(device = args.device)
        self.codebook = Codebook(args).to(device = args.device)
        self.quant = nn.Embedding(args.latent_dim, 1).to(device = args.device)  
        self.post_quant = nn.Embedding(args.latent_dim, 1).to(device = args.device)
            
    def forward(self, embedded_data = None): 
        encoded_text = self.encoder(embedded_data)
        codebook_mapping, codebook_indices, q_loss = self.codebook(encoded_text)
        decoded_text = self.decoder(codebook_mapping)

        return decoded_text, codebook_indices, q_loss

    ## Transformer에 사용되는 것들

    def encode(self, embedded):
        encoded_text = self.encoder(embedded)
        quant_encoded_text = self.quant(encoded_text)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_encoded_text)

        return codebook_mapping, codebook_indices, q_loss
    
    def decode(self, z):
        post_quant_mapping = self.post_quant(z)
        decoded_text = self.decoder(post_quant_mapping)
        return decoded_text
    
    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        lambda_ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lambda_ = torch.clamp(lambda_, 0, 1e4).detach()

        return 0.8*lambda_
    
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value = 0.):
        if i < threshold:
            disc_factor = value
        return disc_factor
        
    def load_checkpoint(self,path):
        self.load_state_dict(torch.load(path))



