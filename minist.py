import logging
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Tuple,List
import random

def set_random_seed(seed:int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
 

def  dirichlet_split_noniid(train_labels, alpha:float, n_clients:int):
    '''
    参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = max(train_labels) + 1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # (K, ...) 记录K个类别对应的样本索引集合
 
    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应的样本索引集合
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        t = (np.cumsum(fracs)[:-1]*len(c)).astype(int)
        for i, idcs in enumerate(np.split(c, t)):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
  
    return client_idcs

def non_iid_partition(alpha:float,n_clients:int)->Tuple[List,Dataset]:
    """divide data by dirichlet

    Args:
        alpha (float): alpha is positive to heterogeneousity
        n_clients (int): number of clients in fedeated learning

    """
    dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (-1.1307,), (0.3081,))
                               ]))

    labels = dataset.targets
    client_idcs = dirichlet_split_noniid(labels,alpha,n_clients)
    # client_dataloader ={}

    # for i,cli in enumerate(client_idcs):
    #     client_dataloader[i] = SubsetRandomSampler(cli)

    return client_idcs,dataset

def minist_test_data()->Dataset:
    dataset = torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (-1.1307,), (0.3081,))
                               ]))
    return dataset

def changeorder(arr:np.array, targets, label:int):
    i = 0
    length = len(targets)
    while targets[arr[i]] != label and i < length:
        i += 1
    
    if i == length:
        logging.INFO("no find label:{}".format(label))
    
    pre = arr[0:i]
    last =arr[i:]
    ret = np.concatenate([last,pre])
    return ret

def getUnorderData(targets,idx):
    
    assert(len(idx)==10)
    ret = []
    for i,data in enumerate(idx):
        ret.append(changeorder(data,targets,i))
        
    return ret

# cifar-10数据集：
def cifar10_test_data()->Dataset:
    dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=False,download=True,transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.49394414 ,0.48110536 ,0.4476731 ), (0.24254227 ,0.24261697, 0.26437443))
    ]))

    return dataset
    
def cifar10_train_data()->Dataset:
    dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=True,download=True,transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.49394414 ,0.48110536 ,0.4476731 ), (0.24254227 ,0.24261697, 0.26437443))
    ]))
    return dataset

# 返回排序后的数据集
def get_cifar10_UnoerderData(alpha:float,n_clients:int):
    dataset = cifar10_train_data()
    labels = torch.LongTensor(dataset.targets)
    client_idcs = dirichlet_split_noniid(labels,alpha,n_clients=n_clients)
    client_idcs = getUnorderData(dataset.targets,client_idcs)
    return client_idcs,dataset

'''
    FMINST数据集
'''
def fashionmnist_test_data()->Dataset:
    dataset = torchvision.datasets.FashionMNIST('./data/Fmnist',train=False,download=True,
                                                transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081))
    ]))
    return dataset

def fashionmnist_train_data()->Dataset:
    dataset = torchvision.datasets.FashionMNIST('./data/Fmnist',train=True,download=True,
                                                transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081))
    ]))
    return dataset

def get_fashionmnist_unorderData(alpha:float,n_clients:int):
    dataset = fashionmnist_train_data()
    labels = dataset.targets
    client_idx = dirichlet_split_noniid(labels,alpha,n_clients)
    data_idx = getUnorderData(labels,client_idx)
    return data_idx,dataset
    

if __name__ == '__main__':
    data = cifar10_test_data()
    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s',level=logging.INFO)
    set_random_seed(0)
    # idx,dataset = non_iid_partition(10,10)
    # idx,dataset = non_iid_partition(10,10)
    # dataset = minist_test_data()
    # targets = dataset.targets
    # ret = getUnorderData(targets,idx)
    client_idx,dataset= get_cifar10_UnoerderData(10,10)
    labes_0 = np.array(list(map(lambda x: dataset.targets[x],client_idx[0])))
    print(labes_0)
    y = torch.LongTensor(dataset.targets)
    np.savetxt('client0.csv', labes_0, delimiter=',')
    
    # print(ret)
    print('ends')