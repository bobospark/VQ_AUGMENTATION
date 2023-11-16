'''
VQ_Augmentation 전체 진행 Main

Generator를 덜 학습시키기 위해, Generator를 학습시키는 횟수를 줄임

'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
from tqdm import tqdm
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
import torch.autograd as autograd


from VQ_Model import VQAugmentation
from discriminator import Discriminator
# from training_dataset import EmbeddingDataset 
from utils import seed_everything

import random
from random import seed

from accelerate import Accelerator
from utils import greedy_search

# ----------
#  Training 1
# ----------

class train_VQAugmentation(nn.Module):
    def __init__(self, args):
        super(train_VQAugmentation, self).__init__()

        self.model = VQAugmentation(args).to(device = args.device)
        self.discriminator = Discriminator(args).to(device = args.device)  # normal distribution으로 initialize
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)



        # Loss weight for gradient penalty
        self.lambda_gp = 100   #### original default is 10
        # self.prepare_training()  # 저장공간 만들기
        with open(f'{args.data_path}/{args.model_name}_embeddings.pkl', "rb") as f:
            self.data = pickle.load(f)

        self.train(args)

    def configure_optimizers(self, args): 
        vq_lr = args.vq_lr
        opt_vq = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()) +
            list(self.model.codebook.parameters()),
            lr = vq_lr, eps = 1e-8, betas = (args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr = vq_lr, eps = 1e-8, betas = (args.beta1, args.beta2))
        
        return opt_vq, opt_disc

    def train(self, args):

        seed_everything(args)

        dataloader = DataLoader(self.data, batch_size=args.batch_size, shuffle=True,)  

        self.model = self.model.to(device = args.device)
        self.discriminator = self.discriminator.to(device = args.device)

        steps_per_epoch =  args.batch_size  # len(train_dataset)  

        self.lambda_gp = 100

        self.criterion = torch.nn.CrossEntropyLoss()

        Tensor = torch.cuda.FloatTensor 

        # start a new wandb run to track this script
        if args.active_log:
            import wandb
            wandb.init(
                # set the wandb project where this run will be logged
                project=f"VQ-Augmentation-Generator-{args.model_name}-renewal-ver2",
                entity = "bobos_park",
                name = f'Params_epochs_{args.n_epochs}_batch_{args.batch_size}_vqlr_{args.vq_lr}_num_code_{args.num_codebook_vectors}',
                reinit=True
                # track hyperparameters and run metadata
            )
            wandb.config = {
                "architecture": args.model_name,
                # "dataset": args.dataset,
                "epochs": args.n_epochs,
                "learning_rate": args.vq_lr,
                "num_codebook_vectors": args.num_codebook_vectors,
                "seed": args.seed,
                }

        # # ---------------------
        # #  Train VQGAN 
        # # ---------------------

        for epoch in range(args.n_epochs):
            with tqdm(range(len(dataloader))) as pbar:
                for i, embs in zip(pbar, dataloader): 

                    embs = embs.to(device = args.device)
                    embs = embs.type(Tensor) 

                    # Sample noise and labels as generator input
                    # 여기서부터 모델 seed 고정 필요. 
                    decoded_text, _, q_loss = self.model(embs) 
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    disc_real = self.discriminator(embs) 
                    disc_fake = self.discriminator(decoded_text)

                    disc_factor = self.model.adopt_weight(args.disc_factor, epoch*steps_per_epoch + i, threshold = args.disc_start)  # 많이 진행되면 gan_loss를 0으로 만들기 위한 factor

                    disc_loss = -torch.mean(disc_real) + torch.mean(disc_fake)  # WGAN Loss

                    self.opt_disc.zero_grad()  # discriminator의 gradient를 0으로 초기화
                    disc_loss.backward(retain_graph = True)  # discriminator의 loss를 backpropagation
                    self.opt_disc.step()  # discriminator의 weight를 update

                    self.opt_vq.zero_grad()  # generator의 gradient를 0으로 초기화
                    if i % args.n_critic == 0:
                    # ---------------------
                    #  Train Generator
                    # ---------------------
                        decoded_text, _, q_loss = self.model(embs)  # q_loss : codebook vector와 encoded latent vector의 L2 distance * 2

                        disc_fake = self.discriminator(decoded_text)

                        rec_loss = torch.abs(embs - decoded_text).mean()

                        cosine_loss = (1- F.cosine_similarity(embs, decoded_text)).mean()

                        cosine_rec_loss =  args.rec_loss_factor * rec_loss  + args.cosine_loss_factor * (cosine_loss) 

                        g_loss = -torch.mean(disc_fake)  # generator의 loss. GAN의 loss와 반대로 만들어야 한다.

                        vq_loss = cosine_rec_loss + q_loss + disc_factor * g_loss

                        vq_loss.backward()  # generator의 loss를 backpropagation
                        self.opt_vq.step()  # generator의 weight를 update

                    pbar.set_postfix(
                        Epoch = f'{epoch+1}/{args.n_epochs}',
                        cosie_loss = np.round(cosine_loss.cpu().detach().numpy().item(), 5),
                        q_loss = np.round(q_loss.cpu().detach().numpy().item(), 5),
                        g_loss = np.round(g_loss.cpu().detach().numpy().item(), 5),
                        VQ_Loss = np.round(vq_loss.cpu().detach().numpy().item(), 5),  # Generator Loss
                        GAN_Loss = np.round(disc_loss.cpu().detach().numpy().item(), 3),  # Discriminator Loss
                    )
                    pbar.update(0)
                    if args.active_log:
                        wandb.log({
                                   "VQ_loss": vq_loss, 
                                   "DISC_loss" : disc_loss, 
                                   "g_loss": g_loss, 
                                   "cosine_loss" : cosine_loss, 
                                   "rec_loss" : rec_loss,
                                   "cosine_rec_loss" :cosine_rec_loss,
                                   "q_loss": q_loss 
                                   })

            # Save the model
            if not os.path.exists(f'/workspace/cache_VQ_model_ver2'):
                os.mkdir(f'/workspace/cache_VQ_model_ver2')

            with open( '/workspace/cache_VQ_model_ver2/%s_%s_%s.pkl'%(args.model_name, args.vq_lr, args.num_codebook_vectors), 'wb') as f:
                pickle.dump(self.model, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")
    parser.add_argument("--vq_lr", type=float, default=2e-05, help="adam: learning rate(default : 0.00002)")
    parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss scalar (default : 0.25), codebook vector에 사용")
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    
    parser.add_argument("--num-codebook-vectors", type = int , default = 2048, help = "number of codebook vectors (default : 512)")
    parser.add_argument("--latent-dim", type = int , default = 128, help = "codebook dimension (default : 512)")

    parser.add_argument("--disc-start", type=int, default=50, help="epoch to start discriminator training (default : 0)")
    parser.add_argument("--disc-factor", type=float, default=1.0, help="Discriminator loss scalar (default : 0.1)")
    parser.add_argument("--rec-loss-factor", type=float, default=1.0, help="rec loss scalar (default : 1)")
    parser.add_argument("--cosine-loss-factor", type=float, default=1.0, help="mse loss scalar (default : 1)")
    parser.add_argument("--n-critic", type=int, default=3, help="number of training steps for discriminator per iter")
    
    parser.add_argument("--few-shot-type", type=str, default='finetune', help="finetune | prompt")
    parser.add_argument("--seed", type=int, default=42, help="13 | 21 | 42 | 87 | 100")

    parser.add_argument("--embedded-size", type=int, default=1024, help="input length for Transformer, i.e. 128, 256, 512")

    parser.add_argument("--model-name", type=str, default='bert-large', help="roberta-large | bert-large-uncased")
    parser.add_argument("--device", type=str, default='cuda', help="cuda | cpu")
    
    parser.add_argument("--text-shape", type=int, default=1024, help="1024 | 2048 | 4096") 

    parser.add_argument("--active-log", type=bool , default=True, help="True | False")  # wandb log를 남길지 말지 True면 남김, False면 남기지 않음
    parser.add_argument("--greedy", type=bool , default=True, help="True | False")  # greedy search를 할지 말지 True면 greedy search, False면 beam search

    parser.add_argument('--data-path', type = str, default = '/workspace/dataset', help = 'data path [dataset | dataset]')

    args = parser.parse_args()

    greedy_search(args.greedy, args, 'VQ_run', train_VQAugmentation)

