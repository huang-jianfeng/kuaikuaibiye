from abc import abstractmethod
import torch
import numpy as np
import copy 
import logging
from typing import  Dict, List
from  client import Client
import torch.nn as nn
import wandb
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import combinations
from scipy.special import comb

class Server:
    
    def __init__(self,clis:List[Client], model:nn.Module, config:Dict,test_dataset:Dataset=None,):
        logging.info("server init")
        self.clis = clis 
        self.global_model = model
        self.test_loss=[]
        self.test_dataset = test_dataset
        self.config = config
        self.next_index=0 #下一轮应该使用的索引
        self.fractions = [1]*len(self.clis)
        
    
    
    def update_model(self,updated_models:List[Dict],weights:List)->Dict:
        """aggragate"""
        # logging.info("========server:start aggragate=============")
        ret = {} 
        total_weight = sum(weights)
        for key in updated_models[0].keys():
            ret[key] = 0
        
            for i,w in enumerate(weights):
                ret[key] += updated_models[i][key] * (w/total_weight)
            
        return ret
                

    def get_global_model(self):
        """get current model"""
    
    def set_client(self,index:int,cli:Client):

        self.clis[index] = cli
    
    def step_sampling_clients(self,num_clients:int, sampling_step:int):
        assert(len(self.clis)==10)
        l = len(self.clis)
        cli_idx = []
        cur_index = self.next_index
        for i in range(num_clients):
            
            cli_idx.append(cur_index)
            
            cur_index += sampling_step
            cur_index = cur_index % 10
        
        self.next_index += 1
        self.next_index = self.next_index % 10
        
        sampling_clis=[]
        logging.info("server:sampling clinents:")
        logging.info(str(cli_idx))
        for idx in cli_idx:
            sampling_clis.append(self.clis[idx])
        return sampling_clis 
        
    def sampling_clients(self, num_clients:int)->List[Client]:
        """sampling nun_clients clients from total clients

        Args:
            num_clients (int): the number of sampling clients

        Returns:
            List[Client]: clients are sampled
        """
        l = len(self.clis)
        cli_idx = np.random.choice(l,num_clients,False)
        sampling_clis=[]
        logging.info("server:sampling clinents:")
        logging.info(str(cli_idx))
        for idx in cli_idx:
            sampling_clis.append(self.clis[idx])
        return sampling_clis 
        
    def start(self):
        logging.info("server:start federated learning:")
        cli_update_period = (self.config['rounds']-180)//4
        for round in range(self.config['rounds']):
            logging.info("server:========+++======round[{0}] start ============:".format(round))
            cur_model_dict_list=[]
            cli_sample_num=[]
            # sampling_clis = self.sampling_clients(self.config['sampling_num'])
            sampling_clis = self.step_sampling_clients(self.config['sampling_num'],1)
            # sampling_clis = self.select_clis(self.clis,self.config['sampling_num'])
            # sampling_clis = self.select_clis_weight(self.clis,self.config['sampling_num'])

            # if (round+1) % cli_update_period == 0:
            for cli in self.clis:
                   cli.add_data(0.003) 

            for cli in sampling_clis:
                cli.set_model_dict(copy.deepcopy(self.global_model.state_dict()))
            
                cli.train()
                cur_model_dict_list.append(cli.get_model_dict())
                cli_sample_num.append(cli.get_num_sample())
                # logging.info(cli.get_model_dict()) 
                # logging.info(cli.counter)
                    
            updateed_model_dict = self.update_model(cur_model_dict_list,cli_sample_num)
            self.global_model.load_state_dict(updateed_model_dict,strict=True)
            
            if (round+1) % self.config['test_frq'] == 0:
                acc,test_loss = self.test_on_test_dataset()
                logging.info("round={1},test_loss = {0},accuracy={2}".format(test_loss,round,acc))
                
                train_acc,train_loss = self.test_on_all_clients()
                wandb.log({"server_loss_on_test":test_loss,"server_acc_on_test":acc,
                           "train_loss":train_loss,"train_acc":train_acc})
            logging.info("server:##################round[{0}] end #################".format(round))

    def test_on_all_clients(self):

        accs=[]
        losses=[]
        cli_sample_num=[]
        for cli in self.clis:
                cli.set_model_dict(copy.deepcopy(self.global_model.state_dict()))
                acc,loss = cli.test_on_trainset()
                accs.append(acc)
                losses.append(loss)
                cli_sample_num.append(cli.get_num_sample())
                
        avg_loss = sum(losses)/sum(cli_sample_num)
        avg_acc = sum(accs)/sum(cli_sample_num)
        self.test_loss.append(avg_loss)
        
        
        return avg_acc,avg_loss
  
        # wandb.log({"test_on_all_clients/loss":avg_loss})
        # np.save('lossresult-{0}'.format(round),np.array(self.test_loss))
                
    def test_on_test_dataset(self):
        test_data_loader = DataLoader(self.test_dataset,self.config['batch_size_test'])
        correct = 0
        test_loss = 0
        model = self.global_model
        lossfunc = nn.CrossEntropyLoss(reduction='sum') 
        model.eval()
        with torch.no_grad():
            for x,target in test_data_loader:
                output = model(x)
                los = lossfunc(output,target)
                test_loss  = test_loss + los.item()
                pred = output.data.max(1,keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        correct = correct/len(self.test_dataset)
        test_loss /= len(self.test_dataset)
        # logging.info("client[{0}] finish test.".format(self.id))
        return correct,test_loss
    
    
    def compute_sim(self,ids):

        sim = 0.0
        for a,b in combinations(ids,2):
            x = self.clis[a].counter 
            y = self.clis[b].counter 
            sim += x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return sim

        """
        选择k个client，被选中的clients两两之间相似度的和应该最小。
        """
    def select_clis(self,clients:List[Client],k:int):
        
        sampled_clis = None 
        ret = []
        min_sim = float('inf')
        n = len(clients) 
        sim_mat = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                x = clients[i].counter
                y = clients[j].counter
                norm2 = np.linalg.norm(x)*np.linalg.norm(y)
                if norm2 ==0:
                    sim_mat[i][j] = 0
                else:
                    sim_mat[i][j] = x.dot(y)/norm2
        for com in combinations(list(range(n)),k):
            tmp_sim = 0
            for two in combinations(com,2):
                tmp_sim += sim_mat[two[0]][two[1]]
            if tmp_sim < min_sim:
                sampled_clis = com
                min_sim = tmp_sim
         
                
        for id in sampled_clis:
            ret.append(clients[id]) 
        logging.info("seleced clients:{}".format(sampled_clis))
        return ret
    
    def select_clis_weight(self,clients:List[Client],k:int):
        
        n = len(clients) 
        sim_mat = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                x = clients[i].counter
                y = clients[j].counter
                norm2 = np.linalg.norm(x)*np.linalg.norm(y)
                if norm2 ==0:
                    sim_mat[i][j] = 0
                else:
                    sim_mat[i][j] = x.dot(y)/norm2

        min_sim = float('inf')
        selected=None
        for com in combinations(list(range(len(clients))),k):
            tmp_sim = 0
            for two in combinations(com,2):
                tmp_sim += clients[two[0]].distrubution_sim*clients[two[1]].distrubution_sim*sim_mat[two[0]][two[1]]*self.fractions[two[0]]*self.fractions[two[1]]
            
            if tmp_sim < min_sim:
                selected = com           
                min_sim = tmp_sim
        ret = [] 
        for idx in selected:
            ret.append(clients[idx])
            self.fractions[idx] += 0.002
        
        logging.info("seleced clients:{}".format(selected))
        return ret