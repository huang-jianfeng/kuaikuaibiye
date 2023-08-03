import logging
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import copy
import torch
import torch.nn.functional as F
from typing import List,Tuple,Dict


class Client:

    def __init__(self,id:int,model:nn.Module,dataset:Data.Dataset,
                 data_idx:List,config:Dict,test_dataset:Data.Dataset ):
        logging.info("client[{0}] init".format(id),)
        self.model = model
        # sampler = SubsetRandomSampler(data_idx)
        # self.dataloader = Data.DataLoader(dataset,config['batch_size'],shuffle=False,sampler=sampler,drop_last=False)
        self.dataset = dataset
        self.config = config
        self.id = id

        self.total_num_sample = len(data_idx)
        self.total_data_idx = data_idx

        self.cur_data_idx_fraction = 0
        self.num_sample = 0
        self.data_idx =[] 
        
        self.test_dataset = test_dataset

        counter_cap = config['counter_cap'] 
        self.counter_cap = counter_cap
        self.counter=np.zeros([self.counter_cap])
        
        self.newcounter = np.zeros([self.counter_cap]) 
        self.distrubution_sim=0

        self.bound = 0
    
    def one_data_arrive(self,pos):
        label = self.dataset[pos][1]
        self.newcounter[label%self.counter_cap] += 1
        
        

    def update_data_idx(self,inc_idx:List):
        self.newcounter= np.zeros([self.counter_cap])
        
        for idx in inc_idx:
           self.one_data_arrive(idx)
        norms = np.linalg.norm(self.counter)*np.linalg.norm(self.newcounter)
        if  norms== 0:
            self.distrubution_sim = 0
        else:
            self.distrubution_sim = self.newcounter.dot(self.counter)/norms
        if sum(self.newcounter) == 0:
            self.distrubution_sim = 1
        self.data_idx.extend(inc_idx)
        self.counter = self.counter + self.newcounter
            
    def add_data(self,inc:float):

        diff = int(inc *self.total_num_sample)
        new_bound = min(diff+self.num_sample,self.total_num_sample)
        self.update_data_idx(self.total_data_idx[self.num_sample:new_bound])
        self.num_sample = new_bound 
        

    def train(self):
        
        logging.info(":=============client[{0}] start trainning==================:".format(self.id))
        
        optimizer = torch.optim.SGD(self.model.parameters(),lr=self.config['lr'],momentum=self.config['momentum'],weight_decay=self.config['weight_decay'])
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        sampler = SubsetRandomSampler(self.data_idx)
        dataloader = DataLoader(self.dataset,self.config['batch_size_train'],shuffle=False,sampler=sampler)
        total_loss = 0
        for epoch in range(self.config['epoch']):
            # logging.info("client[{0}] epoch={1}".format(self.id,epoch))
            # sum = 0
            for batch_idx,(data,target) in  enumerate(dataloader):
                # plt.imshow(data[0][0],cmap='gray', interpolation='none')
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (batch_idx+1) % self.config['client_frq_print'] == 0:
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           self.num_sample,
                                                                           100. * batch_idx / len(dataloader),
                                                                           loss.item()/len(target)))
            
            # logging.info('epoch = {0},loss={1}'.format(i,loss.item()))
            # logging.info("client[{0}] total samplers = {1}".format(self.id,len()))
            # losses.append(loss.item())
        
        logging.info(":===client[{0}] finish trainning  =====:".format(self.id))
    
    def test_on_trainset(self)->Tuple[float,float]:
        """ 
        test on data using model

        Returns:
            Tuple[float,float]: correct,sum of loss for all data in this client
        """
        logging.info(":=======client[{0}] start test on train set.===========:".format(self.id))
        correct = 0
        test_loss = 0
        self.model.eval()
        sampler = SubsetRandomSampler(self.data_idx)
        dataloader = DataLoader(self.dataset,self.config['batch_size_test'],shuffle=False,sampler=sampler)
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():
            for x,target in dataloader:
                output = self.model(x)
                loss = criterion(output,target)
                test_loss  = test_loss + loss.item()
                _, predicted = torch.max(output.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
                correct += (predicted == target).sum().item()

        logging.info("client[{0}] finish test: loss={1}/t acc={2}/{3}.".format(self.id,test_loss/self.num_sample,correct,self.num_sample))
        return correct,test_loss
    
    def test_on_test_dataset(self):
        
        correct = 0
        test_loss = 0
        self.model.eval()
        
        dataloader = DataLoader(self.test_dataset,self.config['batch_size_test'])
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():
            for x,target in dataloader:
                output = self.model(x)
                loss = criterion(output,target)
                test_loss  = test_loss + loss.item()
                _, predicted = torch.max(output.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
                correct += (predicted == target).sum().item()
                
        num_of_sample = len(self.test_dataset)
        logging.info("client[{0}] finish test: loss={1}\t acc={2}/{3}."
                     .format(self.id,test_loss/num_of_sample,correct,num_of_sample))
        return correct/num_of_sample,test_loss/num_of_sample
    
        

    def data_update_inc(self,inc:float=0.0):

        if(self.cur_data_idx_fraction+inc > 1):
            self.cur_data_idx_fraction = 1.0
        else:
            self.cur_data_idx_fraction += inc 
        high_bound = int(len(self.total_data_idx)*self.cur_data_idx_fraction)
        if high_bound > self.total_num_sample:
            high_bound = self.total_num_sample
        self.num_sample = high_bound
        self.data_idx = self.total_data_idx[0:high_bound]
        
    def set_data_idx(self,idx:List[int]):
        self.data_idx = idx
        
    def get_model_dict(self):
        
        return self.model.state_dict()
    
    def set_model_dict(self,model_dict:dict):
        self.model.load_state_dict(model_dict)
    

    def get_num_sample(self)->int:
        
        return self.num_sample 
        