import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataloader import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import random
import time
import math
import os
import pickle
import numpy as np
import shutil
#device='cuda:0'
device='cuda:0'
import copy

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,output_dim,batch_size):
        super(Model, self).__init__()
        self.batch_size=batch_size
        self.fc1 = nn.Linear(input_dim*2,hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.fc3=nn.Linear(hidden_dim2, output_dim)
        self.fc4 = nn.Linear(output_dim*2,output_dim)
        self.fc5 = nn.Linear(output_dim,int(output_dim/4))
        self.fc6=nn.Linear(int(output_dim/4), 2)
        self.ac=nn.LeakyReLU()
    def forward(self, x):
        x = self.fc1(x.view(x.shape[0], -1))  # 将输入转换为[bs, input_dim, seq_len]形状
        x =self.ac(x)
        x = self.fc2(x)  
        x =self.ac(x)
        x = self.fc3(x) 
        x =self.ac(x)
        return x

    def match(self, x):
        x = self.fc4(x)  # 将输入转换为[bs, input_dim, seq_len]形状
        x =self.ac(x)
        x = self.fc5(x)  
        x =self.ac(x)
        x = self.fc6(x) 
        x =self.ac(x)
        return x

class Trainer:
    def __init__(self, data_loader, model, optimizer, device, batch_size,data,num_epochs):

        self.data_loader = data_loader
        self.model = model

        self.optimizer = optimizer
        self.device = device
        self.batch_size=batch_size
        self.data=data
        self.num_epochs=num_epochs
       
        #self.scheduler=scheduler
        self.best_acc=None
        model.to(device)
    
        
    def train(self,epoch):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        data_loader = self.data_loader
        train_set=data_loader.train_set   
        # criterion 
        criterion = torch.nn.CrossEntropyLoss()
    
        total_accuracy = 0
        total_loss = 0
        optimizer.zero_grad()
        self.train_data_sampler = self.data_loader.train_data_sampler(batch_size=self.batch_size,neg_rate=1)
        n_batch = len(self.train_data_sampler)
        dataset_size = self.train_data_sampler.get_dataset_size()
        real_dataset_size = dataset_size
        #breakpoint()
        for i_b, batch in tqdm(enumerate(self.train_data_sampler), total=n_batch):
            #breakpoint()
            triples = [i[0] for i in batch]
            batch_size_ = len(batch)
            real_triples = [ i[0] for _, i in enumerate(batch)]
            #构建labels，trainset
            labels=[i[1] for i in batch]
            labels=torch.tensor(labels).to(device=device)
            head_subgraph_seqs,tail_subgraph_seqs=data_loader.load_subgraph_embedding(train_set, real_triples)
            head_seq_feature=model(head_subgraph_seqs.to(device))
            tail_seq_feature=model(tail_subgraph_seqs.to(device))
            preds=model.match(torch.cat((head_seq_feature, tail_seq_feature),dim=-1)).to(device=device).squeeze(1)
            loss=criterion(preds,labels)
            loss.backward()
            optimizer.step()
            pred_labels=preds.argmax(dim=1)
            total_accuracy+=(pred_labels==labels).int().sum().item()
            total_loss+=loss.item()*batch_size_
            torch.save(model,'./model_{0}'.format(epoch))
        if epoch%5==0:
            print('train-epc:',epoch,'loss:',total_loss/real_dataset_size,'accuracy:',total_accuracy/real_dataset_size)    


    @torch.no_grad()
    def test(self,epoch):
        model = self.model
        optimizer = self.optimizer

        device = self.device
        data_loader = self.data_loader

        train_set=data_loader.train_set
        test_data_sampler = self.data_loader.test_data_sampler(batch_size=self.batch_size,neg_rate=1)

        # criterion 
        criterion = torch.nn.CrossEntropyLoss()

        model.eval()
        total_test_loss = 0
        total_test_acc=0
        
        n_batch = len(test_data_sampler)
        dataset_size = test_data_sampler.get_dataset_size()
        
        for i_b, batch in tqdm(enumerate(test_data_sampler), total=n_batch):
            triples = [i[0] for i in batch]
            batch_size_ = len(batch)
            labels=[i[1] for i in batch]
            labels=torch.tensor(labels).to(device=device)
            head_subgraph_seqs,tail_subgraph_seqs=data_loader.load_subgraph_embedding(train_set,triples)
            head_seq_feature=model(head_subgraph_seqs.to(device))
            tail_seq_feature=model(tail_subgraph_seqs.to(device))
            preds=model.match(torch.cat((head_seq_feature, tail_seq_feature),dim=-1)).to(device=device)
            loss=criterion(preds,labels)
            pred_labels=preds.argmax(dim=1)
            total_test_acc+=(pred_labels==labels).int().sum().item()
            total_test_loss+=loss.item()*batch_size_

        print('test-epc:',epoch,'loss:',total_test_loss/dataset_size,'accuracy:',total_test_acc/dataset_size)
        return total_test_acc


    def train_loop(self):
        for epoch in range(self.num_epochs):
            self.train(epoch)
            self.test(epoch)


input_dim =768*2*2 # Define your input dimension based on the embedding size
hidden_dim1 =768*2# Define your hidden dimension
hidde_dim2=768
output_dim = 512# Define your output dimension
learning_rate =0.01 # Define your learning rate
batch_size = 1024 # Define your batch size
num_epochs = 200# Define your number of epochs
data='wn18rr_cut'
in_paths = {
    'dataset': data,
    'train': '/home/DYP/2023test/baseline/LMKE-main-pri/Ours/data/WN18RR_cut/train_cut.tsv',
    'valid': '/home/DYP/2023test/baseline/LMKE-main-pri/Ours/data/WN18RR_cut/dev.tsv',
    'test': '/home/DYP/2023test/baseline/LMKE-main-pri/Ours/data/WN18RR_cut/test.tsv',
    'text': ['/home/DYP/2023test/baseline/LMKE-main-pri/Ours/data/WN18RR_cut/my_entity2text.txt', 
        '/home/DYP/2023test/baseline/LMKE-main-pri/Ours/data/WN18RR_cut/relation2text.txt']
}
# data='fb15k-237'     
# in_paths = {
#     'dataset': data,
#     'train': './data/fb15k-237/train.tsv',
#     'valid': './data/fb15k-237/dev.tsv',
#     'test': './data/fb15k-237/test.tsv',
#     'text': ['./data/fb15k-237/FB15k_mid2description.txt', 
#         #'./data/fb15k-237/entity2textlong.txt', 
#         './data/fb15k-237/relation2text.txt']
# }
data_loader=DataLoader(in_paths,batch_size,neg_rate=1,rdrop=0)
# Define model, loss function, and optimizer
model = Model(input_dim, hidden_dim1,hidde_dim2, output_dim,batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)
trainer=Trainer(data_loader, model, optimizer, device,batch_size,data,num_epochs)
trainer.train_loop()