import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from loguru import logger
import tensorflow as tf
from torch.utils.data import DataLoader,Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import multiprocessing
import traceback

def collate_fn_pad(batch):
    """collate_fn
    Pads batch of different length
    :param batch: batch
    """
    
    sequences = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    
    padded_seqs = pad_sequence(sequences, batch_first=True)
    lengths = torch.LongTensor(lengths)
    labels  = torch.LongTensor(labels)
    
    return padded_seqs, labels, lengths


class UninumTrainDataset(Dataset):
    """UninumDataset
    create train data object
    """
    def __init__(self, hparams):
        # load in data
        self.hparams = hparams
        self.data_file = os.path.join(self.hparams.train_dir,'train_seqs.npz')
        
        data = np.load(self.data_file)
        self.X = data['x']
        self.y = data['y']
        
    def __getitem__(self, idx):
    
        X = self.X[idx]
        y = self.y[idx]
        length = len(X)
        
        X = torch.tensor(X)
        
        return X, y, length
  
    def __len__(self):
        return len(self.X)
        


class UninumTestDataset(Dataset):
    """UninumTestDataset
    create train data object
    """
    def __init__(self, hparams):
        # load in data
        self.hparams = hparams
        self.data_file = os.path.join(self.hparams.test_dir,'test_seqs.npz')
        
        data = np.load(self.data_file)
        self.X = data['x']
        self.y = data['y']
        
    def __getitem__(self, idx):
                  
        X = self.X[idx]
        y = self.y[idx]
        length = len(X)
        
        X = torch.tensor(X)
        
        return X, y, length
  
    def __len__(self):
        return len(self.X)
        