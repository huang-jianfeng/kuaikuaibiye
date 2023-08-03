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
    
    test_dataset = minist_test_data()

    wandb.init(
        
    # set the wandb project where this run will be logged
    project="uesct_fedarated_learning",
    
    # track hyperparameters and run metadata
    config=config
    
    )
    cli = Client(id=0,model = Net(),dataset=dataset,data_idx=data_idx[0],config=config,test_dataset=test_dataset)
    
    cli_update_period = (500-180)//4
    
    cli.add_data(0.02)
    
    for i in range(500):
        logging.info("############rounds {0}############".format(i))
        if (i+1) % cli_update_period == 0:
            cli.data_update_inc(0.003)
        
        cli.train()
        
        if (i+1) % config['test_frq'] == 0:
            acc,test_loss = cli.test_on_test_dataset()
            # num = cli.get_num_sample()
            # acc,test_loss = acc/num,test_loss/num
            wandb.log({'test_loss':test_loss,'test_acc':acc})
            logging.info("client test: loss={},acc={}".format(test_loss,acc))
    
    # for i in range(config['clientnum']):
    #     cli = Client(id = i,model=Net(),dataset=dataset,data_idx=data_idx[i],config=config)
    #     clis.append(cli)
    
    # test_dataset = minist_test_data()
    
    # server = Server(clis=clis,model=Net(),config=config,test_dataset=test_dataset)
    # server.start()
