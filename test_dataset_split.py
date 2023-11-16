'''
VQ-Generator Train 코드

Token화 된 데이터셋을 Few-shot으로 불러오는 코드

#Few_Shot class
#MyDataset class
#make_roberta_dataset

'''


import random
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



class Testset_split(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data = _Set_Dataset_to_Token(args)  # test_dataset2 -> Mydataset

        # Set dataset
        data_file_path = '%s/%s_dataset.pkl'%(args.data_path, args.dataset)
        with open(data_file_path, "rb") as f:
            self.dataset = pickle.load(f)
        self.processor = processors_mapping[args.dataset]
        # Define the path to the saved model file
        roberta_model_file_path = '%s/%s_class_%s.pkl'%(args.model_path, args.model_name, args.num_classes)

        if args.dataset not in ['mnli', 'mnli-mm']:
            Testset_split.validation_length = self.dataset['validation'].num_rows
            Testset_split.test_length = self.dataset['test'].num_rows
        if args.dataset in ['mnli', 'mnli-mm']:
            if args.dataset == 'mnli':
                Testset_split.validation_length = self.dataset['validation_matched'].num_rows
                Testset_split.test_length = self.dataset['test_matched'].num_rows
            if args.dataset == 'mnli-mm':    
                Testset_split.validation_length = self.dataset['validation_mismatched'].num_rows
                Testset_split.test_length = self.dataset['test_mismatched'].num_rows

        # Load the model from the file using pickle
        with open(roberta_model_file_path, "rb") as f:
            self.model_roberta = pickle.load(f).to(device = args.device)

            
    def forward(self, args, train_eval = None, iter = None):
        random.seed(args.seed)

        if args.dataset not in ['mnli', 'mnli-mm']:
            if train_eval == 'validation' or train_eval == 'val_test':
                self.train_eval = 'validation'
            else:
                self.train_eval = train_eval
        if args.dataset in ['mnli', 'mnli-mm']:
            if args.dataset == 'mnli':
                if train_eval == 'validation' or train_eval == 'val_test':
                    self.train_eval = 'validation_matched'
                if train_eval == 'test':
                    self.train_eval = 'test_matched'
            if args.dataset == 'mnli-mm':
                if train_eval == 'validation' or train_eval == 'val_test':
                    self.train_eval = 'validation_mismatched'
                if train_eval == 'test':
                    self.train_eval = 'test_mismatched'

        self.data = self.dataset[self.train_eval]  # train_eval =[ 'train', 'validation', 'test']

        if train_eval == 'val_test':
            self.splited_data = self.data[iter*args.val_test_batch:(iter+1)*args.val_test_batch]
            tensor_index_total = self.data['idx'][iter*args.val_test_batch:(iter+1)*args.val_test_batch]

        if train_eval in ['test', 'test_matched', 'test_mismatched']:
            self.splited_data = self.data[iter*args.val_test_batch:(iter+1)*args.val_test_batch]
            tensor_index_total = self.data['idx'][iter*args.val_test_batch:(iter+1)*args.val_test_batch]

        self.data_encoded, self.text = self.load_data.forward(args, datasets = self.splited_data)  # , self.train_eval

        temp_tensor = []

        for i in range(len(self.data_encoded)):
            temp_tensor.append(({'input_ids':self.data_encoded[i][0]['input_ids'], 'attention_mask':self.data_encoded[i][0]['attention_mask']},self.data_encoded[i][1]))
        #### 위에가 Few_Shot.py ###

        self.embeddings = []
        self.attention_mask_ = []
        self.labels = []

        for token in temp_tensor:
            input_ids = token[0]['input_ids'].to(device = args.device)  # (batch_size, seq_len)
            attention_mask = token[0]['attention_mask'].to(device = args.device)  # (batch_size, seq_len)
            label = token[1].detach().cpu().numpy()
            
            with torch.no_grad():
                embedded_data = self.model_roberta.get_input_embeddings()(input_ids)  # <- 이거가 다른거였음 그냥
                embeddings = embedded_data.detach().cpu().numpy()
            
            self.embeddings.append(torch.tensor(embeddings))
            self.attention_mask_.append(torch.tensor(attention_mask).long())
            self.labels.append(torch.tensor(label).long())


        self.embeddings = torch.cat(self.embeddings, dim = 0).reshape(len(self.embeddings), args.max_seq_length, args.text_shape).to(device = args.device)  # 67은 max_length로 바꿔야함 
        self.attention_mask = torch.cat(self.attention_mask_, dim = 0).reshape(len(self.embeddings), args.max_seq_length).to(device = args.device)
        
        return self.embeddings, self.attention_mask ,self.labels
        ### 위에가 make_augmentation.py ###

class val_test_dataset(Dataset):
    def __init__(self, args, train_eval = None, iter = None):

        self.args = args
        self.load_data = Testset_split(args)

        self.embeddings, self.attention_mask, self.labels = self.load_data.forward(args, train_eval = train_eval, iter = iter)

        self.data_ = {'inputs_embeds': self.embeddings, 'attention_mask': self.attention_mask}
        self.data_label = self.labels


    def __getitem__(self, index):

        return {'inputs_embeds': self.data_['inputs_embeds'][index], 'attention_mask':self.data_['attention_mask'][index]}, self.data_label[index]

    def __len__(self):
        
        return len(self.data_label)
    