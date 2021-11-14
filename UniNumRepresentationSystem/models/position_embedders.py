import torch
import torch.nn as nn
import math
import pdb
import numpy as np


class PositionEmbedding(nn.Module):
    """
    [batch_size, sequence_length, embedding_dimension]
    """
    
    def __init__(self, max_sequence_length : int, emb_dim : int):
        super(PositionEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_sequence_length, emb_dim, padding_idx = None)
        
    def forward(self, input_embeddings : torch.tensor):
        
        positions = torch.arange(input_embeddings.shape[1], device = input_embeddings.device)
        return input_embeddings + self.embedding(positions)
    
"""Create standard positional embedding: https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, dropout : float, max_length : int):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x) 
    
