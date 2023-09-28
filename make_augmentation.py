#--------------------------------------------------------------
#  Few-Shot으로 Trained 된 VQ-Augmentation을 통해
#  Class당 16개의 Augmentation된 데이터셋을 만드는 코드
#--------------------------------------------------------------

import torch
import torch.nn as nn
import pickle


from Few_Shot_data import Few_Shot

from torch.utils.data.dataloader import DataLoader

## RoBERTa Few-shot Learning 시 Augmentation Dataset을 만드는 코드
class MakeAugmentation(nn.Module):
    def __init__(self, args):
        super(MakeAugmentation, self).__init__()
        self.args = args
        torch.manual_seed(args.seed)

        # Define the path to the saved model file
        roberta_model_file_path = '%s/%s_class_%s.pkl'%(args.model_path, args.model_name, args.num_classes)

        # Load the model from the file using pickle
        with open(roberta_model_file_path, "rb") as f:
            self.model_roberta = pickle.load(f).to(device = args.device)
        
        #  'mrpc', 'qqp', 'stsb', 'mnli', 'mnli-mm', 'qnli', 'rte', 'wnli', 
        if args.data_augmentation :
            vq_model_file_path = '%s/%s_%s_%s.pkl'%(args.vq_model_path, args.model_name, args.vq_lr, args.num_codebook_vectors)

            with open(vq_model_file_path, "rb") as f:
                self.model_vq = pickle.load(f).to(device = args.device)

    def forward(self, args, train_eval = None, ):  # num_classes = None, num_shots = None

        dataset = Few_Shot(args)  # from test_data.py

        samples, indexes, classes = dataset.forward(args, train_eval = train_eval)
        
        # index 리스트를 파일에 저장
        # index_file_path = 'cache_index/seed_%s_indexes.pkl'%(args.seed)
        # with open(index_file_path , 'wb') as f:
        #     pickle.dump(indexes, f)
        self.embeddings = []
        self.attention_mask_ = []
        self.labels = []

        for token in samples:
            input_ids = token[0]['input_ids'].to(device = args.device)  # (batch_size, seq_len)
            attention_mask = token[0]['attention_mask'].to(device = args.device)  # (batch_size, seq_len)
            label = token[1].detach().cpu().numpy()
            
            with torch.no_grad():
                embedded_data = self.model_roberta.get_input_embeddings()(input_ids)  # <- 이거가 다른거였음 그냥
                embeddings = embedded_data.detach().cpu().numpy()
            
            self.embeddings.append(torch.tensor(embeddings))
            self.attention_mask_.append(torch.tensor(attention_mask).long())
            self.labels.append(torch.tensor(label).long())
                
        ##### 여기까지가 Train data set Embedding으로 만드는 부분 #####


        ##### 여기서부터는 Train Augmentation data set 만드는 부분 #####

        if train_eval == 'train':  # eval의 경우 Augmetation이 필요 없으므로 Pass
            if args.data_augmentation :
                self.model_vq.eval()
                augmented_data = []
                total_augmented_data = []
                ## 문장들을 하나씩 뽑아서 Augmentation을 진행한다.
                for embs, label in zip(self.embeddings, self.labels):
                    embs = embs.to(device = args.device)
                    label = label.to(device = args.device)

                    decoded_sentence = []
                    with torch.no_grad():
                        # 단어들을 하나씩 뽑아서 학습 진행
                        for j in range(embs.shape[0]):
                            text = embs[j,:].view(1 , -1) 
                            label = label.view(-1)
                            decoded_text = self.model_vq(text, label)[0]  # min_encoding_indices와 q_lossㄴ는 빼고 뽑음
                            decoded_sentence.append(decoded_text)
                        augmented_data = torch.cat(decoded_sentence, dim = 0) # [67, 1024]
                        total_augmented_data.append(augmented_data)

                total_augmented_data_set = torch.cat(total_augmented_data, dim = 0).reshape(len(total_augmented_data), args.max_seq_length, args.text_shape)
                self.embeddings = torch.cat(self.embeddings, dim = 0).reshape(len(self.embeddings), args.max_seq_length, args.text_shape).to(device = args.device)
                self.attention_mask = torch.cat(self.attention_mask_, dim = 0).reshape(len(self.embeddings), args.max_seq_length).to(device = args.device)
                
                return self.embeddings, total_augmented_data_set, self.attention_mask ,self.labels

            # Few-shot learning을 위한 data augmentation을 하지 않는다면

            else:  # Conventional Few-shot learning cell
                self.embeddings = torch.cat(self.embeddings, dim = 0).reshape(len(self.embeddings), args.max_seq_length, args.text_shape).to(device = args.device)  # 67은 max_length로 바꿔야함 
                self.attention_mask = torch.cat(self.attention_mask_, dim = 0).reshape(len(self.embeddings), args.max_seq_length).to(device = args.device)               
                _ = []
                
                return self.embeddings, _, self.attention_mask ,self.labels
            
        if train_eval != 'train':  # For Evaluation Cell
            self.embeddings = torch.cat(self.embeddings, dim = 0).reshape(len(self.embeddings), args.max_seq_length, args.text_shape).to(device = args.device)  # 67은 max_length로 바꿔야함 
            self.attention_mask = torch.cat(self.attention_mask_, dim = 0).reshape(len(self.embeddings), args.max_seq_length).to(device = args.device)
            _ = []
            return self.embeddings, _, self.attention_mask ,self.labels
        