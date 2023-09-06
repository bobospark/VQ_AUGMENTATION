# import EarlyStopping
# from pytorchtools import EarlyStopping
import torch
import numpy as np
import os

from transformers import set_seed
import random

# from VQ_run import train_VQAugmentation

# from Finetuning_RoBERTa import Test_training

def seed_everything(args):
    seed = args.seed
    set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed is set to %s'%seed)
    print()


class EarlyStopping:
    """주어진 patience 이후로 validation acc가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=10, verbose=False, delta=0, path= None):
        """
        Args:
            patience (int): validation acc가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation acc의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
        self.val_acc_max = 0.0

        self.delta = delta
        self.path = path

    def __call__(self, model, eval_acc):
        self.eval_acc = eval_acc
        # acc = val_acc
        if self.best_acc is None:
            self.best_acc = self.eval_acc
            self.save_checkpoint(self.eval_acc, model)

        elif self.eval_acc < self.best_acc :  #- self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            # accuracy 기준으로 저장
            if self.eval_acc > self.best_acc:
                self.save_checkpoint(self.eval_acc, model)

            # accuracy에 변동이 없지만 loss가 크게 감소하면 저장

            self.best_acc = self.eval_acc
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''validation acc가 증가하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
            print(f'model eval acc:{self.eval_acc}')
        if not os.path.exists(f'checkpoint_roberta_model'):
            os.mkdir(f'checkpoint_roberta_model')
        torch.save(model.state_dict(), self.path)  # state_dict()
        self.val_acc_max = val_acc


def greedy_search(greedy_or_not, args, model_name, model):
    num_of_training = 0
    datasets = ['cola', 'sst-2', 'qqp', 'mrpc']  # 'mnli', 'mnli-mm', 'qnli', 'rte', 'wnli', 

    seeds = [13, 21, 42, 87, 100]
    num_codebook_vectors = [512, 1024, 2048]
    vq_lrs = [1e-5, 2e-5, 5e-5]

    if model_name == 'VQ_run':
        if greedy_or_not:
            for dataset in datasets:
                for seed in seeds:
                    for num_codebook_vector in num_codebook_vectors:
                        for vq_lr in vq_lrs:
                            num_of_training += 1
                            args.dataset = dataset
                            args.seed = seed
                            args.num_codebook_vectors = num_codebook_vector
                            args.vq_lr = vq_lr
                            print(num_of_training, '/', len(datasets)*len(seeds)*len(num_codebook_vectors)*len(vq_lrs))
                            model(args)
        if not greedy_or_not:
            model(args)

    if model_name == 'Finetuning_RoBERTa':
        augments = [True,False]  # True
        if greedy_or_not:
            epochs = [150, 200]
            lrs = [1e-5, 2e-5, 5e-5]
            for augment in augments:
                args.data_augmentation = augment
                if args.data_augmentation:
                    for epoch in epochs:
                        for dataset in datasets:  # 2개
                            for seed in seeds:  # 5개
                                for lr in lrs:  # 3개
                                    for vq_lr in vq_lrs:  # 3개
                                        for num_codebook_vector_ in num_codebook_vectors: # 3개

                                            num_of_training+=1
                                            args.epochs = epoch
                                            args.dataset = dataset                                            
                                            args.seed = seed  # [13, 21, 42, 87, 100]
                                            args.lr = lr  # [1e-5, 2e-5,5e-5]
                                            args.vq_lr = vq_lr  # [1e-5, 2e-5, 5e-5]
                                            args.num_codebook_vectors = num_codebook_vector_  # [512, 1024, 2048]
                                            args.vq_model_path = f'cache_VQ_model/cache_for_{args.dataset}_renewal' 
                                            print(dataset, num_of_training, '/', len(num_codebook_vectors)*len(vq_lr)*len(lr)*len(seed))
                                            model(args)

                if not args.data_augmentation:
                    print('data augmentation False')
                    i = 0
                    for epoch in epochs:  # 2개
                        for dataset in datasets:  
                            for seed in seeds:  # 5개
                                for lr in lrs:  # 3개
                                    i+=1
                                    args.epochs = epoch
                                    args.dataset = dataset
                                    args.seed = seed
                                    args.lr = lr
                                    print(dataset, num_of_training, '/', len(seed)*len(lrs))
                                    model(args)
        
        if not greedy_or_not:
            for augment in augments:
                args.data_augmentation = augment
                model(args)