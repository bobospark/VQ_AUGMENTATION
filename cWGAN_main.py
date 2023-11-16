import argparse
import os
import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torchvision.utils import save_image
from torch.utils.data import DataLoader

from cWGAN_model import Generator, Discriminator
import json

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import pickle
from training_dataset import make_dataset

from utils import seed_everything, greedy_search

# def sample_image(n_row, batches_done):
#     """Saves a grid of generated digits ranging from 0 to n_classes"""
#     # Sample noise

#     cuda = True

#     Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#     LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

#     z = Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))
#     # Get labels ranging from 0 to n_classes for n rows
#     labels = np.array([num for _ in range(n_row) for num in range(n_row)])
#     with torch.no_grad():
#         labels = LongTensor(labels)
#         gen_imgs = generator(z, labels)
#     save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    
    Tensor = torch.cuda.FloatTensor # if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor # if cuda else torch.LongTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

class EmbeddingDataset(Dataset):
    def __init__(self, opt):
        
        ####################### 바꿀부분 #######################
        ## Data Loading ##
        # with open('cache_embeddings/%s_%s_%s_%s.json'%(opt.model_name_or_path, opt.few_shot_type, opt.task_name, opt.seed)) as json_file:
        #     data = json.load(json_file)
        # with open(f'/workspace/task_dataset/{opt.dataset}_dataset.pkl', 'rb') as f:
        #     data = pickle.load(f)
        dataset = make_dataset(opt, train_eval = 'train')



        self.embeddings = []
        self.labels = []

        self.num_classes = 0

        for data in dataset:
            # for emb_i in range(len(data[i])):
            self.embeddings.append(data[0]['inputs_embeds'].squeeze(0))
            self.labels.append(data[1].long())
            if self.num_classes == 0 and data[1] == torch.tensor(0):
                self.num_classes +=1
            if self.num_classes == 1 and data[1] == torch.tensor(1):
                self.num_classes += 1
            if self.num_classes == 2 and data[1] == torch.tensor(2):
                self.num_classes += 1

        # self.num_classes = len(data)
        self.data_len = len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index].float(), self.labels[index]

    def __len__(self):
        return self.data_len


class GAN_Training():
    def __init__(self, opt):
        super(GAN_Training, self).__init__()

        self.opt = opt
        self.train(opt)
    
    def train(self, opt):
    
        print('seed 고정')
        seed_everything(opt)

        embeddings_set = EmbeddingDataset(opt)

        dataloader = DataLoader(embeddings_set, batch_size=8, shuffle=True)
    

        if opt.active_log:
            import wandb
            wandb.init(
                # set the wandb project where this run will be logged
                project = f'GAN-Generator-{opt.model_name}',
                entity = "bobos_park",
                name = f'{opt.model_name}_{opt.few_shot_type}_{opt.dataset}_{opt.seed}',
                reinit = True,
                # Set entity to specify your username or team name
                # ex: entity='carey',
                # Set the job type to distinguish different runs
                job_type = 'train',
                # Track hyperparameters and run metadata
                config = opt,
            )
            wandb.config = {
                "architecture": opt.model_name,
                "dataset": opt.dataset,
                "epochs": opt.n_epochs,
                "learning_rate": opt.lr,
                "seed": opt.seed,
            }



        emb, lbl = embeddings_set.__getitem__(0)

        opt.n_classes=embeddings_set.num_classes
        opt.img_shape = emb.shape

        cuda = True if torch.cuda.is_available() else False

        # Loss weight for gradient penalty
        lambda_gp = 100   #### original default is 10

        # Initialize generator and discriminator
        generator = Generator(opt).to(device = opt.device)
        discriminator = Discriminator(opt).to(device = opt.device)

        # if cuda:
        #     generator.cuda()
        #     discriminator.cuda()


        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        batches_done = 0
        for epoch in range(opt.n_epochs):
            for i, (embs, labels) in enumerate(dataloader):
                batch_size = embs.shape[0]

                # Move to GPU if necessary
                real_embs = embs.type(Tensor)
                labels = labels.type(LongTensor)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise and labels as generator input
                z = Tensor(np.random.normal(0, 1, (embs.shape[0], opt.latent_dim)))

                # Generate a batch of embeddings
                fake_embs = generator(z, labels)

                # Real embeddings
                real_validity = discriminator(real_embs, labels)
                # Fake embeddings
                fake_validity = discriminator(fake_embs, labels)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(
                                    discriminator, real_embs.data, fake_embs.data,
                                    labels.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % opt.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of embeddings
                    fake_embs = generator(z, labels)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake embeddings
                    fake_validity = discriminator(fake_embs, labels)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                    )
                    if opt.active_log:
                        wandb.log({
                                   "DISC_loss" : d_loss, 
                                   "g_loss": g_loss, 
                                   })

                    batches_done += opt.n_critic

        if not os.path.exists('/workspace/cache_gan_generators'):
            os.mkdir('/workspace/cache_gan_generators')
        torch.save(generator.state_dict(), '/%s/cache_gan_generators/%s_%s_%s_%s.pth'%('workspace',opt.model_name_or_path, opt.few_shot_type, opt.dataset, opt.seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=3, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--dataset", type=str, default='sst-2', help="dataset to use")
    # model_args.model_name_or_path, model_args.few_shot_type, data_args.task_name, training_args.seed)
    parser.add_argument("--model_name_or_path", type=str, default='roberta-large', help="roberta-larger | bert-large-cased")
    parser.add_argument("--few_shot_type", type=str, default='finetune', help="finetune | prompt")
    # parser.add_argument("--task_name", type=str, default='sst-2', help="sst-5 | cola | mrpc | ...")
    parser.add_argument("--seed", type=int, default=42, help="13 | 21 | 42 | 87 | 100")
    parser.add_argument("--seq_len", type=int, default=128, help="input length for Transformer, i.e. 128, 256, 512")

    parser.add_argument("--device", type = str, default = 'cuda:0', help = 'Number of GPU')
    parser.add_argument("--model_name", type = str, default = 'roberta-large', help = 'Base model (default : roberta-large)')
    parser.add_argument("--data_augmentation", type = bool, default = False, help = 'Data augmentation (default : True)')
    parser.add_argument('--model-path', type = str, default = f'{os.getcwd()}/model_file', help = 'model path [model_file | model_file]')
    parser.add_argument('--data-path', type = str, default = f'{os.getcwd()}/task_dataset', help = 'data path [data_file | data_file]')
    parser.add_argument("--num-shots", type=int, default=16, help="number of shots per class 16")  # train에서 data 종류 확인한 후 설정
    parser.add_argument("--max-seq-length", type=int, default=128, help="128 | 256 | 512")
    parser.add_argument('--data-loader-type', type = str, default = 'few-shot', help = 'few shot | prompt')
    parser.add_argument("--text-shape", type=int, default=1024, help="1024 | 2048 | 4096") # train에서 data 종류 확인한 후 설정
    parser.add_argument("--greedy", type=bool , default = True, help="True | False")  # greedy search를 할지 말지 True면 greedy search, False면 beam search
    parser.add_argument("--active_log", type=bool , default = True, help="True | False")  # greedy search를 할지 말지 True면 greedy search, False면 beam search

    opt = parser.parse_args()

    print(opt)

    greedy_search(opt.greedy, opt, 'GAN', GAN_Training)

    

