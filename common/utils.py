import os
import torch
import random
import numpy as np
import yaml
from types import SimpleNamespace

def load_config(config_path):
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Recursively convert dict to Namespace
    def dict_to_namespace(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)

    return dict_to_namespace(config_dict)


def setup_seed(random_seed, cudnn_deterministic=True):
    """ set_random_seed(random_seed, cudnn_deterministic=True)

    Set the random_seed for numpy, python, and cudnn

    input
    -----
      random_seed: integer random seed
      cudnn_deterministic: for torch.backends.cudnn.deterministic

    Note: this default configuration may result in RuntimeError
    see https://pytorch.org/docs/stable/notes/randomness.html
    """

    # # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False


def read_metadata(dir_meta, is_eval=False):
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()
    
    if (is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list


def read_metadata_itw(dir_meta):
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        key,_,label = line.strip().split()
         
        file_list.append(key)
    return file_list

def read_mosdata(dir_meta, file_id):
    with open(dir_meta, 'r') as f:
        metadata = {}
        for line in f:
            key, mos = line.strip().split()
            metadata[key] = float(mos)

    filtered_meta = {key: metadata[key] for key in file_id if key in metadata}
    
    return filtered_meta