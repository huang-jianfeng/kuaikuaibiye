from client import  Client
 
from model import Net
from minist import non_iid_partition
from minist import set_random_seed
from config import get_config
from minist import minist_test_data

import logging
if __name__ == '__main__':
    model = Net()
    data_index_dict,dataset = non_iid_partition(10,1)

    set_random_seed(0)
    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s',level=logging.INFO)
    clientnum=5
    config = get_config('config.yaml')
    testdataset = minist_test_data()
    
    client = Client(id=1,model=model,dataset=dataset,
                    data_idx=data_index_dict[0],config=config,test_dataset=testdataset)

    client.data_update_inc()
    client.train()
    