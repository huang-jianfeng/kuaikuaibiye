
from model import Net
 
from model import Model
import wandb
from torch.utils.data import Dataset
from minist import set_random_seed
from minist import non_iid_partition
from minist import getUnorderData
from minist import minist_test_data
from client import Client
from fedavg import Server
from config import get_config
import logging
if __name__ =='__main__':
    set_random_seed(10)
    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s',level=logging.INFO)
    config  = get_config('config.yaml')
    clis=[]
    data_idx,dataset = non_iid_partition(config['alpha'],config['clientnum'])
    data_idx = getUnorderData(dataset.targets,data_idx)
    # wandb.init(
        
    # # set the wandb project where this run will be logged
    # project="uesct_fedarated_learning",
    
    # # track hyperparameters and run metadata
    # config=config
    
    # )
        
    test_dataset = minist_test_data()
    for i in range(config['clientnum']):
        cli = Client(id = i,model=Net(),dataset=dataset,data_idx=data_idx[i],config=config,test_dataset=test_dataset)
        cli.add_data(0.02)
        clis.append(cli)

    
    server = Server(clis=clis,model=Net(),config=config,test_dataset=test_dataset)
    server.start()
    wandb.finish()
