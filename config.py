from typing import Dict
import yaml

def get_config(filename:str):

    with open(filename,'r') as f:
        data = yaml.load(f,yaml.FullLoader)
    return data

if __name__ =='__main__':
    data = get_config("config.yaml")
    print(data)
    