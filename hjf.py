import numpy as np
from itertools import combinations
def compute_sim(x,y):

    sim = x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return sim


# Start train and Test --------------------------
if __name__ == '__main__':
    
    x = np.array([5, 5, 5, 5, 5])
    y = np.array([5, 5, 5, 5, 5])
    
    x1 = np.array([10, 10, 10, 10,10])
    y1 = np.array([5,5,5,5,5])
    
    x2 = np.array([100, 100, 100, 100,100])
    y2 = np.array([5,5,5,5,5])
    
    
    print(compute_sim(x,y))
    print(compute_sim(x1,y1))
    print(compute_sim(x2,y2))

    x = np.zeros([10])
    y = np.zeros([10])
    sim = x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y)+float("inf"))
    print(np.linalg.norm(x))
    print(np.linalg.norm(x) == 0)
    