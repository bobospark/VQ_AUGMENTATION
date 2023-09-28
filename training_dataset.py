import torch
from torch.utils.data import Dataset, DataLoader

from random import seed

from make_augmentation import MakeAugmentation

from Few_Shot_data import Few_Shot
import pickle

class make_roberta_dataset(Dataset):  
    def __init__(self, args, train_eval = None):
        
        self.args = args
        self.load_data = MakeAugmentation(args)
        self.train_eval = train_eval

        if train_eval == 'train':
            real_embedding, fake_embedding, attention_mask, labels = self.load_data.forward(args, train_eval = self.train_eval)
        else:  
            real_embedding, _ ,attention_mask, labels  = self.load_data.forward(args, train_eval = self.train_eval)
            # make_roberta_dataset.indexes = indexes

        if args.data_augmentation :
            if self.train_eval == 'train':
                print('Real + Fake(Train)')
                self.total_embedding = torch.cat([real_embedding, fake_embedding], dim=0)
                self.total_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
                self.total_labels = labels + labels
                
                # self.data_ = {'inputs_embeds': self.total_embedding, 'attention_mask': self.total_attention_mask}
                # self.data_label = self.total_labels

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
    

# # VQ_run.py에 쓰임
# class EmbeddingDataset(Dataset):
#     def __init__(self, args, train_eval = None):
#         seed(args.seed)
#         if args.num_classes == 2:
#             # Define the path to the saved roberta_model file
#             model_file_path = '%s/%s_class_%s.pkl'%(args.model_path, args.model_name, args.num_classes)

        
#         # Load the roberta_model from the file using pickle
#         with open(model_file_path, "rb") as f:
#             self.roberta_model = pickle.load(f)
#         self.roberta_model = self.roberta_model.to(device = args.device)

#         ## Few-shot Learning Dataset
#         dataset = Few_Shot(args)  # from test_data.py

#         samples, indexes, classes = dataset.forward(args, train_eval = train_eval, num_classes = args.num_classes, num_shots = args.num_shots)

#         # 사용되는 indexes는 일정하게 저장됨

#         # if not os.path.exists('cache_index'):
#         #     os.mkdir('cache_index')
#         # with open('cache_index/%s_%s_%s_%s_%s.pkl'%(args.model_name, 'index',args.few_shot_type, args.dataset, args.seed), 'wb') as f:
#         #     pickle.dump(indexes, f)

#         data_loader = DataLoader(samples, batch_size = args.batch_size, drop_last=True,shuffle = True , )  # 여기서 왜 Dataloader로 한번 섞고 있찌?
#         data = {}
#         for class_ in classes:
#             data[class_] = []

#         for i, embedding in enumerate(data_loader):
#             input_ids = embedding[0]['input_ids'].to(device = args.device)  # (batch_size, seq_len)
#             attention_mask = embedding[0]['attention_mask'].to(device = args.device)  # (batch_size, seq_len)
#             # token_type_ids = token_type_ids.to(device = args.device)
#             labels = embedding[1].detach().cpu().numpy()
            
#             with torch.no_grad():
#                 for j in range(args.batch_size):
#                     embedded_data = self.roberta_model.roberta.embeddings(input_ids[j].view(1, -1)) 
                    
#                     embeddings = embedded_data.detach().cpu().numpy()

#                     for k in classes:
#                         if labels[j] == k:
#                             data[k].append(embeddings)


#         self.embeddings = []  # len = args.num_shots * args.num_classes
#         self.labels = []
#         ## dictionary to list
#         for i in data:  # i = 0 or 1
#             for emb_i in range(len(data[i])):
#                 self.embeddings.append(torch.tensor(data[i][emb_i]).squeeze(0))
#                 self.labels.append(torch.tensor(int(i)).long())

#         self.num_classes = len(data)
#         self.data_len = len(self.labels)


#     def __getitem__(self, index):
#         return self.embeddings[index].float(), self.labels[index]

#     def __len__(self):
#         return self.data_len