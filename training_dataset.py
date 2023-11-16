import torch
from torch.utils.data import Dataset, DataLoader

from random import seed

from make_augmentation import MakeAugmentation

from Few_Shot_data import Few_Shot
import pickle

class make_dataset(Dataset):  
    def __init__(self, args, train_eval = None):
        
        self.args = args
        self.load_data = MakeAugmentation(args)
        self.train_eval = train_eval

        if train_eval == 'train':
            real_embedding, fake_embedding, attention_mask, labels = self.load_data.forward(args, train_eval = self.train_eval)
        else:  
            real_embedding, _ ,attention_mask, labels  = self.load_data.forward(args, train_eval = self.train_eval)

        if args.data_augmentation :
            if self.train_eval == 'train':
                print('Real + Fake(Train)')
                self.total_embedding = torch.cat([real_embedding, fake_embedding], dim=0)
                self.total_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
                self.total_labels = labels + labels
                

            if not self.train_eval == 'train':
                if train_eval == 'validation':
                    print('Real + Fake(Validation)')
                if train_eval == 'val_test':
                    print('Real + Fake(Validation_Test)')
                if train_eval == 'test':
                    print('Real + Fake(Test)')

                self.total_embedding = real_embedding
                self.total_attention_mask = attention_mask
                self.total_labels = labels 
                
            self.data_ = {'inputs_embeds': self.total_embedding, 'attention_mask': self.total_attention_mask}
            self.data_label = self.total_labels

        else:
            if self.train_eval == 'train':
                print('Only Real(Train)')
            if self.train_eval == 'validation':
                print('Only Real(Validation)')
            if self.train_eval == 'val_test':
                print('Only Real(Validation_Test)')
            if self.train_eval == 'test':
                print('Only Real(Test)')

            self.data_ = {'inputs_embeds': real_embedding, 'attention_mask': attention_mask}
            self.data_label = labels

    def __getitem__(self, index):

        return {'inputs_embeds': self.data_['inputs_embeds'][index], 'attention_mask':self.data_['attention_mask'][index]}, self.data_label[index]

    def __len__(self):
        
        return len(self.data_label)
    
