# from client import  Client
# from model import Net
from minist import non_iid_partition
from minist import set_random_seed
from config import get_config
from minist import minist_test_data
from minist import getUnorderData
from matplotlib import pyplot as plt


if __name__ == '__main__':
    set_random_seed(10)
    # logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s',level=logging.INFO)
    config  = get_config('config.yaml')
    clis=[]
    data_idx,dataset = non_iid_partition(config['alpha'],config['clientnum'])
    data_idx = getUnorderData(dataset.targets,data_idx)
    
    test_dataset = minist_test_data()
    
    data = data_idx[1]
    labels = dataset.targets.numpy()[data]
    
    threshold= 0.8
    miu = 0.0 #分布均值
    sum =0.0 #误差累积
    points = [] #变化点   

    
    total_sum=0.0
    cnt = 0
    
    for index,label in enumerate(labels):
        y = float(label)
        sum += y-miu
        
        if sum > threshold or sum < -threshold:
            points.append(index)
            cnt = 1
            total_sum = label
            miu = label
            sum=0
        else:
            cnt += 1
            total_sum += label
            miu = total_sum/cnt
        

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(labels)),labels,label='label')
    ax.vlines(points, 0, 10, linestyles='dashed', colors='red',label='change-point')
    plt.legend()
    plt.show()