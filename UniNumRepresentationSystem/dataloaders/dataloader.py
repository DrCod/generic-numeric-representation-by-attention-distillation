import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import numpy as np

class NumericalDataset(Dataset):
    """NumericalDataset object
    Args:
       data_path : path to data csv file
    """
    def __init__(self,
                data_path : str = None,
                data_aux_path :  str = None,
                k : int = 7
                ):
        
        self.data_path = data_path
        self.data_aux_path = data_aux_path
        self.k = k
        
        assert os.path.splitext(data_path)[-1] == '.csv', "Invalid file format."
        
        self.data = list(pd.read_csv(data_path)[['text', 'target']].to_dict('index').values())
        aux_data = list(pd.read_csv(data_aux_path)[['source', 'target']].to_dict('index').values())
        
        # randomly sample top k
        self.rand_aux_data = random.choice(aux_data, size = k)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            
            item = self.data[idx]
            
            text = item['text']
            label = item['target']
            
            aux_tuples = [(item['source'], item['target']) for item in self.rand_aux_data]
            
            return {
                'text' : text,
                'label' : torch.tensor(label).float(),
                'length' : len(text),
                'topk' : aux_tuples
            }
        
        
def NumericalDatasetCollator(batch):
    
    X =  [b['text'] for b in batch]
    X_length = [len(x) for x in X]
    y =  [b['label'] for b in batch]
    aux_x = [x[0] for x in b['topk'] for b in batch]
    aux_y = [x[1] for x in b['topk'] for b in batch]
    
    y = torch.tensor(y, dtype = torch.float)
    
    return (X, y, aux_x, aux_y)