'''
VQ-Generator Train 코드

Token화 된 데이터셋을 Few-shot으로 불러오는 코드

#Few_Shot class
#MyDataset class
#make_roberta_dataset

'''


import random
from random import seed
from random import randint
from torch.utils.data import Dataset

import pickle
import torch
import numpy as np
from dataset_processor import processors_mapping
# from tokenizing import Set_Dataset_to_Token
from torch.utils.data.dataloader import DataLoader

def sum_text(texts1, texts2, sep_token):
    full_text = np.array(texts1,dtype=object) + np.array([sep_token for _ in range(len(texts1))])+np.array(texts2,dtype=object)

    return list(full_text)


class _Make_pair(Dataset):
    def __init__(self, args, encodings, labels):
        self.args = args
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, index):
        # get the encoded input data for the given index
        if self.args.data_loader_type == 'few-shot':
            encoding = {key: tensor[index] for key, tensor in self.encodings.items()}

        # get the corresponding label for the given index
        label = torch.tensor(self.labels[index])

        return encoding, label
    
    def __len__(self):
        return len(self.labels)


class _Set_Dataset_to_Token():  
    def __init__(self, args):
        super(_Set_Dataset_to_Token, self).__init__()

        tokenizer_file_path = '%s/%s_%s.pkl'%(args.model_path, args.model_name, 'tokenizer')
        # Load the model from the file using pickle
        with open(tokenizer_file_path, "rb") as f:
            self.tokenizer = pickle.load(f)


    def forward(self, args, datasets = None):
        self.datasets = datasets
        features = [i for i in self.datasets.keys()]

        # if args.dataset not in ['qqp', 'mnli', 'mnli-mm'] or train_eval not in  ['val_test', 'test']:
        if len(features) == 3:#  sst2, cola # args.dataset == 'sst2' or args.dataset == 'cola':
            linetext, label, idx =  self.datasets[features[0]], self.datasets[features[1]] ,self.datasets[features[2]]
            encodings = self.tokenizer(linetext, 
                                    add_special_tokens=True, 
                                    padding= 'max_length',  # 'max_length'
                                    max_length= args.max_seq_length,  # args.max_seq_length
                                    truncation=True,
                                    return_attention_mask=True, 
                                    return_tensors='pt')  # 왜인지 encoded_plus를 쓰면 버그뜸...
            
        if len(features)==4: # mrpc, qqp, mnli(class 3), qnli, rte, wnli   #args.dataset == 'mrpc' or args.dataset == 'qqp' or args.dataset == 'rte' or args.dataset == 'wnli':
            linetext1, linetext2, label, idx =  self.datasets[features[0]], self.datasets[features[1]], self.datasets[features[2]] ,self.datasets[features[3]]
            linetext = sum_text(linetext1, linetext2, self.tokenizer.sep_token)
            
            encodings = self.tokenizer(linetext, 
                                    add_special_tokens=True, 
                                    padding= 'max_length',  # 'max_length'
                                    max_length= args.max_seq_length,  # args.max_seq_length
                                    truncation=True,
                                    return_attention_mask=True, 
                                    return_tensors='pt')

        self.data = _Make_pair(args, encodings, labels = label)

        return self.data, linetext

    def __len__(self):
        return len(self.data)   
    
    # def __getitem__(self, idx):
    #     text, label = self.data[idx]
    #     input_ids = self.tokenizer.encode(text, add_special_tokens=True)
    #     attention_mask = [1] * len(input_ids)
    #     return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(label)



class Few_Shot(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data = _Set_Dataset_to_Token(args)  # test_dataset2 -> Mydataset

        # Set dataset

        data_file_path = '%s/%s_dataset.pkl'%(args.data_path, args.dataset)
        with open(data_file_path, "rb") as f:
            self.dataset = pickle.load(f)
        self.processor = processors_mapping[args.dataset]
        
            
    def forward(self, args, train_eval = None):
        random.seed(args.seed)

        if args.dataset not in ['mnli', 'mnli-mm']:
            if train_eval == 'validation' or train_eval == 'val_test':
                self.train_eval = 'validation'
            else:
                self.train_eval = train_eval
        if args.dataset in ['mnli', 'mnli-mm']:
            if args.dataset == 'mnli':
                if train_eval == 'train':
                    self.train_eval = 'train'
                if train_eval == 'validation' or train_eval == 'val_test':
                    self.train_eval = 'validation_matched'
                if train_eval == 'test':
                    self.train_eval = 'test_matched'
            if args.dataset == 'mnli-mm':
                if train_eval == 'train':
                    self.train_eval = 'train'
                if train_eval == 'validation' or train_eval == 'val_test':
                    self.train_eval = 'validation_mismatched'
                if train_eval == 'test':
                    self.train_eval = 'test_mismatched'

        self.data = self.dataset[self.train_eval]  # train_eval =[ 'train', 'validation', 'test']

        args.num_classes = len(self.processor.get_labels())

        classes = list(range(args.num_classes))

        break_point = 0
        tensor_index = {}

        for class_ in classes:
            tensor_index[class_] = []

        if self.train_eval not in ['test', 'test_matched', 'test_mismatched']:
            while True:
                value = randint(0, self.data.num_rows - 1)
                for class_ in classes:
                    if self.data[value]['label'] == class_:
                        if value not in tensor_index[class_]:
                            if len(tensor_index[class_]) < args.num_shots:
                                tensor_index[class_].append(value)
                                break_point += 1
                            else:
                                pass
                if break_point == args.num_classes * args.num_shots:
                    break
                else:
                    pass

            tensor_index_total = []
            for i in classes:
                tensor_index_total += tensor_index[i]

            if train_eval != 'val_test':  # val_test는 validation에서 쓰이는 데이터 제외한 나머지
                self.few_shot_data = self.data[tensor_index_total]

            if train_eval == 'val_test':
                val_test_idx = [i for i in list(range(len(self.data)-1)) if i not in tensor_index_total]
                self.few_shot_data = self.data[val_test_idx]

        if train_eval in ['test', 'test_matched', 'test_mismatched']:
            self.few_shot_data = self.data[:]
            tensor_index_total = self.data['idx']
 
        self.data_encoded, self.text = self.load_data.forward(args, datasets = self.few_shot_data)  # , self.train_eval

        temp_tensor = []

        for i in range(len(self.data_encoded)):
            temp_tensor.append(({'input_ids':self.data_encoded[i][0]['input_ids'], 'attention_mask':self.data_encoded[i][0]['attention_mask']},self.data_encoded[i][1]))
        
        return temp_tensor, tensor_index_total, classes



    def __len__(self):
        return len(self.data_encoded)
    
