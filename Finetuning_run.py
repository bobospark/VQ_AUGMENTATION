'''
Embedding Vector 다 찾고 난 다음 RoBERTa에 넣어서 학습시키는 코드
'''


import os
import argparse

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn.functional as F

from GPUtil import showUtilization as gpu_usage

from tqdm import tqdm

from sklearn.metrics import accuracy_score

from transformers import set_seed

from utils import EarlyStopping

from torch.utils.data import DataLoader

from make_augmentation import MakeAugmentation
from training_dataset import make_roberta_dataset
from test_dataset_split import Testset_split, val_test_dataset
# from Few_Shot_data import Few_Shot

from testset_maker import testset
# from tokenizing import Set_Dataset_to_Token
from dataset_processor import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping  #  median_mapping

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import pickle

import torch.nn as nn
import numpy as np

from dataset_processor import num_labels_mapping

# os.environ['CUDA_VISIBLE_DEVICE'] = '3'
from utils import seed_everything, greedy_search

class Test_training(nn.Module):
    def __init__(self, args):
        super(Test_training, self).__init__()

        args.num_classes = num_labels_mapping[args.dataset]
        

        # if args.num_classes == 2 :
            # Define the path to the saved model file
        model_file_path = '%s/%s_class_%s.pkl'%(args.model_path, args.model_name, args.num_classes)
        # if args.num_classes == 3 :
        #     # Define the path to the saved model file
        #     model_file_path = '%s/%s-mnli.pkl'%(args.model_path, args.model_name)
        
            # Load the model from the file using pickle
        with open(model_file_path, "rb") as f:
            self.model = pickle.load(f).to(device = args.device)

        # self.tokenizer = Set_Dataset_to_Token(args)
        self.processor = processors_mapping[args.dataset]  # mapping 용으로 쓰임

        self.optim = self.configure_optimizers(args)
        self.load_data = MakeAugmentation(args)
        self.criterion = torch.nn.CrossEntropyLoss()


        self.scheduler = get_linear_schedule_with_warmup(self.optim, 
                                                         num_warmup_steps = args.warmup_steps, 
                                                         num_training_steps = int((args.num_shots * args.num_classes * 2)/args.batch_size)*args.epochs)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr = args.lr, steps_per_epoch = int((args.num_shots * args.num_classes * 2)/args.batch_size), epochs = args.epochs)
        self.testset_split = Testset_split(args)

        early_stopping = EarlyStopping(patience=5)

        self.score = compute_metrics_mapping[args.dataset]

        self.train(args)
        
    def configure_optimizers(self, args):

        opt = AdamW(
                params=self.model.parameters(),
                lr=args.lr,
                # weight_decay=0.01,
                betas = (args.beta1, args.beta2),
                eps=1e-08,
                # amsgrad=False,
                )
        return opt


    def train(self, args):

        print('seed 고정')
        seed_everything(args)
        print('Embedding starts')
        train_set = make_roberta_dataset(args, train_eval = 'train')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,)  
        
        eval_set = make_roberta_dataset(args, train_eval = 'validation')
        # eval_index = make_roberta_dataset.indexes
        eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=True,)
        
        checkpoint = os.path.join(args.save_model, f'{args.model_name}_{args.dataset}_seed_{args.seed}.pt')

        # early_stopping object의 초기화
        early_stopping = EarlyStopping(patience = args.patience, verbose = True, path = checkpoint)

        # wandb 설정해주는 곳
        if args.active_log:
            import wandb
            if args.data_augmentation:
                wandb.init(
                    # set the wandb project where this run will be logged
                    # 1. Conventional Few-shot Classification
                    # 2. VQ-Augmentation-Classification
                    project=  f"VQ-mixture-{args.dataset}-Finetuning-renewal-rate-{args.rate_of_real}",  # epochs-{args.epochs}
                    entity = "bobos_park",
                    name = f'{args.model_name}_main-lr_{args.lr}_vq-lr_{args.vq_lr}_codebook_{args.num_codebook_vectors}_rate_{args.rate_of_real}_seed_{args.seed}',  # _epochs_{args.epochs}
                    reinit=True
                    # track hyperparameters and run metadata
                )
                wandb.config = {
                "architecture": args.model_name,
                "dataset": args.dataset,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "vq-learning_rate": args.vq_lr,
                "num_codebook_vectors": args.num_codebook_vectors,
                "rate of real" : args.rate_of_real,
                "seed": args.seed,
                }
            else:
                wandb.init(
                    # set the wandb project where this run will be logged
                    # 1. Conventional Few-shot Classification
                    # 2. VQ-Augmentation-Classification
                    project=  f"RoBERTa-Conventional-Classification-{args.dataset}-renewal-ver3",  #  #  
                    entity = "bobos_park",
                    name = f'{args.model_name}_{args.dataset}_lr_{args.lr}_seed_{args.seed}',
                    reinit=True
                    # track hyperparameters and run metadata
                )
                wandb.config = {
                "architecture": args.model_name,
                "dataset": args.dataset,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "seed": args.seed,
                }

        # roberta.embeddings freeze
        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False

        set_seed(args.seed)
        

        # Training Start
        best_val_loss = 10000.0
        best_eval_acc = 0.0
        for epoch in range(args.epochs):
            train_losses = []
            print(f'Epoch {epoch + 1}/{args.epochs}')
            with tqdm(range(len(train_loader))) as pbar:
                for batch_idx, (inputs, labels) in enumerate(train_loader):  # 
                    pbar.update(1)

                    outputs = self.model(**inputs)#['last_hidden_state']
                    outputs_prob = torch.softmax(outputs.logits, dim = 1)  # softmax 쓰는건 무조건 맞음
                    labels = labels.to(device=args.device)#view(-1) 

                    train_loss = self.criterion(outputs_prob, labels)
                    train_losses.append(train_loss.item())

                    self.optim.zero_grad()
                    train_loss.backward()
                    self.optim.step()
                    self.scheduler.step()

                    pbar.set_postfix({
                        'train_loss': train_loss.item(), 'epoch': epoch + 1, 'batch': batch_idx + 1, 
                        'train_total_loss': sum(train_losses) / len(train_losses)
                        })
                if args.active_log:
                    wandb.log({"train_total_loss": sum(train_losses) / len(train_losses)})

            print(f'Epoch {epoch + 1}, train loss: {sum(train_losses)  / len(train_losses):.4f}')

            # Evaluation
            with torch.no_grad():
                eval_labels = []
                eval_preds = []
                eval_total_loss = 0
                self.model.eval()
                for batch_idx, (inputs, labels) in tqdm(enumerate(eval_loader)):
                    outputs = self.model(**inputs)  # logits work same as [0] on the back
                    output_prob = torch.softmax(outputs.logits, dim = 1)
                    labels = labels.to(device=args.device)
                    
                    eval_labels.extend(labels.to(device = 'cpu'))
                    eval_preds.extend(output_prob.argmax(dim=1).to(device='cpu'))

                    eval_loss = self.criterion(output_prob, labels)
                    eval_total_loss += eval_loss.item()
                    if args.active_log:
                        wandb.log({"eval_loss": eval_loss.item()})
                # self.scheduler.step()
                avg_eval_loss = eval_total_loss / len(eval_loader)
                print('avg eval loss:', avg_eval_loss)

                eval_acc = accuracy_score(eval_labels, eval_preds)
                print('eval acc:', eval_acc)
                # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
                # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.

                early_stopping(self.model, eval_acc)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if avg_eval_loss < best_val_loss:
                    best_val_loss = avg_eval_loss
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc

                if args.active_log:
                    wandb.log({"avg_eval_loss" : avg_eval_loss,"best_val_loss": best_val_loss, "best_eval_acc": best_eval_acc})
            
            if args.active_log:
                wandb.log({'validation_accuracy': eval_acc})
            
        print('Fine Tuning End')

        print('VAL_Test Start')

        # VAL_Test set 성능 뽑기
        self.model.load_state_dict(torch.load(checkpoint))
        with torch.no_grad():
            val_test_preds = []
            val_test_label = []
            self.model.eval()

            val_test_iter = int(self.testset_split.validation_length/args.val_test_batch) + 1
            with tqdm(range(val_test_iter)) as pbar:
                for i in range(val_test_iter):
                    pbar.update(1)
                    val_test_set = val_test_dataset(args, train_eval = 'val_test', iter = i)
                    val_test_loader = DataLoader(val_test_set, batch_size=200, shuffle=False,)

                    for batch_idx, (inputs, label) in enumerate(val_test_loader):

                        outputs = self.model(**inputs)  # logits work same as [0] on the back
                        val_test_pred_outputs = torch.softmax(outputs.logits, dim = 1)
                        
                        val_test_preds.extend(val_test_pred_outputs.argmax(dim=1).to(device='cpu').tolist())
                        val_test_label.extend(label.to(device = 'cpu').tolist())
                        
                    pbar.set_postfix({
                    'epoch': f'{i + 1}/{val_test_iter}'
                    })
            VAL_TEST_ACC = self.score(args.dataset, np.array(val_test_label), np.array(val_test_preds))

            if args.active_log:
                wandb.log({'VAL_TEST_ACC': list(VAL_TEST_ACC.values())[0]})

            ### Test set 성능 뽑기  ### 
            print('Test Start')
            test_preds = []
            test_idx = []

            test_iter = int(self.testset_split.test_length/args.val_test_batch) + 1
            with tqdm(range(test_iter)) as pbar:
                for i in range(test_iter):
                    pbar.update(1)
                    test_set = val_test_dataset(args, train_eval = 'test', iter = i)
                    test_loader = DataLoader(test_set, batch_size=200, shuffle=False,)
                    self.model.eval()

                    for batch_idx, (inputs, label) in enumerate(test_loader):
                        outputs = self.model(**inputs)  # logits work same as [0] on the back
                        test_pred_outputs = torch.softmax(outputs.logits, dim = 1)
                        test_preds.extend(test_pred_outputs.argmax(dim=1).to(device='cpu').tolist())
                    print(len(test_preds))

            testset(args, test_preds)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQAugmentation")
    # parser = argparse.ArgumentParser(description="VQ_Model")
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train (default: 50)')

    parser.add_argument('--seed', type=int, default=42, help='13 | 21 | 42 | 87 | 100')
    parser.add_argument('--model-name', type = str, default = 'roberta-large', help = 'Base model (default : roberta-large)')
    parser.add_argument('--dataset', type = str, default = 'mnli-mm', help = "Choose the dataset")
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension n_z (default: 256)')

    parser.add_argument("--warmup-steps", type=int, default=0, help= "warmup steps for scheduler")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adamw: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adamw: decay of first order momentum of gradient")
    
    parser.add_argument('--data-loader-type', type = str, default = 'few-shot', help = 'few shot | prompt')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 4) [4 | 6 | 8]')
    parser.add_argument('--val-test-batch', type=int, default=2000, help='Input batch size for training (default: 4) [4 | 6 | 8]')
   
    parser.add_argument('--vq-lr', type=float, default=2e-5, help='Learning rate (default: 0.00001) [1e-5 | 2e-5 | 5e-5]')
    parser.add_argument('--num-codebook-vectors', type=int, default=2048, help='Number of codebook vectors (default: 256)')

    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (default: 0.00001) [1e-5 | 5e-6 | 1e-6]')    
    parser.add_argument("--num-classes", type=int, default=2, help="2 | 3 | 5 | 7 | 10")  # train에서 data 종류 확인한 후 설정
    parser.add_argument("--num-shots", type=int, default=16, help="number of shots per class 16")  # train에서 data 종류 확인한 후 설정
    parser.add_argument("--text-shape", type=int, default=1024, help="1024 | 2048 | 4096") # train에서 data 종류 확인한 후 설정
    parser.add_argument('--num-labels', type = int, default = 2, help = 'number of classification (sst-2 : 2)')
    parser.add_argument('--rate-of-real', type = float, default = 0.0, help = 'mixture rate of the real data, (1-rate is fake rate)')
    parser.add_argument("--max-seq-length", type=int, default=128, help="128 | 256 | 512")

    parser.add_argument("--patience", type=int, default= 100, help="patience for early stopping")

    parser.add_argument('--device', type = str, default = 'cuda:1', help = 'Number of GPU')

    parser.add_argument("--active-log", type=bool , default=True, help="True | False")  # wandb log를 남길지 말지 True면 남김, False면 남기지 않음
    parser.add_argument('--data-augmentation', type = bool, default = True, help = 'Data Augmentation or not [True, False]]')
    parser.add_argument("--greedy", type=bool , default = True, help="True | False")  # greedy search를 할지 말지 True면 greedy search, False면 beam search

    parser.add_argument('--model-path', type = str, default = f'{os.getcwd()}/model_file', help = 'model path [model_file | model_file]')
    parser.add_argument('--vq-model-path', type = str, default = f'{os.getcwd()}/cache_VQ_model_ver2', help = 'vq_model path [model_file | model_file]')
    parser.add_argument('--data-path', type = str, default = f'{os.getcwd()}/task_dataset', help = 'data path [data_file | data_file]')

    parser.add_argument('--save-model', type = str, default = f'{os.getcwd()}/checkpoint_roberta_model', help = 'save model path [model_file | model_file]')
    parser.add_argument('--save-test-result', type = str, default = f'{os.getcwd()}/test_result_ver3', help = 'save model path [model_file | model_file]')
    parser.add_argument('--save-conventional-test-result', type = str, default = f'{os.getcwd()}/test_convention_result_ver3', help = 'save model path [model_file | model_file]')
    
    args = parser.parse_args()

    greedy_search(args.greedy, args, 'Finetuning_RoBERTa', Test_training)
