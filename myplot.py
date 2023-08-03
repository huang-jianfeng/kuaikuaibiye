import numpy as np
import matplotlib.pyplot as plt
from typing import List

def draw(filenames:List[str]):
    for f in filenames:
        data = np.load(f)
        plt.plot(range(len(data)),data)
        
    plt.xlabel('rounds')
    plt.ylabel('y')
    # plt.title(")
    plt.legend()
    # plt.savefig(r'.\lab3\pic\max-min')
    plt.show()
    plt.close()

if __name__ =='__main__':
    files = ['lossresult-339.npy']
    draw(files)