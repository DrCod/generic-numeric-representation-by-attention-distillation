import os
import sys
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import time
import random
from .multi_hash_embedding import MultiHashingEmbedder
from .position_embedders import PositionEmbedding, PositionalEncoding
from .local_self_attention import LocalSelfAttention

device = torch.device("cuda" if torch.cuda.is_available() "cpu")


# MHSA Layer
class MHSA(nn.Module):
    def __init__(self,
                hidden_dim,
                num_attn_heads,
                dropout,
                pos_embedder_type,
                device):
        
        super().__init__()
        
        assert hidden_dim % num_attn_heads == 0, "hidden_dim must be divisible by number of attention heads"
        
        self.hidden_dim = hidden_dim
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = hidden_dim // num_attn_heads
        
        self.Q_fc = nn.Linear(hidden_dim, hidden_dim)
        self.K_fc = nn.Linear(hidden_dim, hidden_dim)
        self.V_fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        if pos_embedder_type == "None":
            self.scale = torch.sqrt(torch.FloatTensor([self.attn_head_size])).to(device)
    
    def forward(self, query, key, value, mask = None):
        
        batch = query.shape[0]

        # query = [bs, query_len, hidden_dim]
        # key =   [bs,   key_len,   hidden_dim]
        # value = [bs, value_len, hidden_dim]
        
        Q = self.Q_fc(query)
        K = self.K_fc(key)
        V = self.V_fc(value)
        
        # Q = [bs, query_len, hidden_dim]
        # K = [bs, key_len, hidden_dim]
        # V = [bs, value_len, hidden_dim]
        
        Q = Q.view(batch_size, -1, self.num_attn_heads, self.attn_head_size)
        K = K.view(batch_size, -1, self.num_attn_heads, self.attn_head_size)
        V = V.view(batch_size, -1, self.num_attn_heads, self.attn_head_size)
        
        # Q = [bs, num_attn_heads, query_len, attn_head_size]
        # K = [bs, num_attn_heads, key_len, attn_head_size]
        # V = [bs, num_attn_heads ,value_len, attn_head_size]
        
        scores = torch.matmul(Q, K.permute(0,1, 3, 2)) / self.scale
        
        # scores = [bs, num_attn_heads, query_len, key_len ]
        
        if mask is not None:
            scores =  scores.masked_fill(mask == 0, -1e10)
            
        attention = torch.softmax(scores, dim = -1)
        
        # attention = [bs, num_attn_heads, query_len, key_len]
        x = torch.matmul(self.dropout(attention), V)
        
        # x = [bs ,n_attn_heads, query_len, attn_head_size]
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # x = [bs, query_len, num_attn_heads, attn_head_size]
        x = x.view(batch_size, -1, self.hidden_dim)
        
        # x = [bs, query_len, hidden_dim]
        
        x = self.fc_o(x)
        
        return x , attention
       
class PositionwiseFeedforward(nn.Module):
    def __init__(self,
                hidden_dim,
                ff_dim,
                dropout,
                activation = 'relu'
                ):
        
        super(PositionwiseFeedforward,).__init__()
        
        self.fc_1 = nn.Linear(hidden_dim, ff_dim)
        self.fc_2 = nn.Linear(ff_, hidden_dim)
        self.activation = activation
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # x = [bs, seq_len, hidden_dim]
        if self.activation == "relu":
            x = self.dropout(torch.relu(self.fc_1(x)))
        elif self.activation == "gelu":
            x = self.dropout(torch.gelu(self.fc_1(x)))
        else:
            ValueError(f"Invalid activation. Expects relu/gelu but received {self.activation}")
        
        # x = [bs, seq_len, ff_dim]
        
        x = self.fc_2(x)
        
        # x = [bs, seq_len, hidden_dim]
        
        return x
        
        
        
        
# Encoder Layer
# The Encoder layer implements the multi-head attention layer that performs the self-attention(MHSA) over the input sequences
class EncoderLayer(nn.Module):
    """
    Sequence-to-Sequence Encoder Layer
    """
    def __init__(self,
                 hidden_dim,
                 num_attn_heads,
                 ff_dim,
                 dropout,
                 window_size,
                 pos_embedder_type,
                 activation,
                 device,
                 use_local_self_attention
                ):
        super(EncoderLayer,).__init__()
        
        if window_size % 2 != 0:
                    raise ValueError('Window size must be an even number so it can be split evenly across previous and' 'following tokens.')
        self.self_attention = LocalSelfAttention(hidden_dim, num_attn_heads, window_size,dropout,auto_pad =False) if use_local_self_attention else MHSA(hidden_dim, num_attn_heads, dropout,pos_embedder_type, device)
        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.ff_ln =  nn.LayerNorm(hidden_dim)
        self.positionwise_ff = PositionwiseFeedforward(hidden_dim, ff_dim, dropout, activation)
        
        self.dropout = dropout
        self.use_local_self_attention = use_local_self_attention
        self.pos_embedder_type = pos_embedder_type
        
        
    def forward(self, src, src_mask):

        # src = [bs, src_len, hidden_dim]
        # src_mask = [ bs, 1, 1, src_len]
        if not self.use_local_self_attention:
            _src, _ = self.self_attention(src, src, src, src_mask)
        else:
            _src =  self.self_attention(src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attention_ln(src + self.dropout(_src))

        # positionwise feedforward
        _src = self.positionwise_ff(src)
        # dropout , residual and layernorm
        src = self.ff_ln(src + self.dropout(_src))

        # src = [bs, sr_len, hidden_dim]

        return src




## Encoder
class Encoder(nn.Module):
    """
    Sequence-to-Sequence Encoder
    """
    def __init__(self,
                input_dim,
                hidden_dim,
                num_layers,
                num_attn_heads,
                ff_dim,
                dropout,
                device,
                max_length,
                embedder_slice_count : int = 0,
                embedder_bucket_count : int = 0,
                token_embedder_type  : str = "hashing",
                pos_embedder_type : str = "canine",
                use_local_self_attention = False,
                activation = "relu"
                ):
        super(Encoder,).__init__()
        
        self.device = device
        
        if activation not in ['relu', 'gelu']: RuntimeError(f"Invalid activation. Expects relu/gelu but got {activation}")
        else:
            self.activation = activation
            
        self.token_embedder_type = token_embedder_type
        self.pos_embedder_type = pos_embedder_type
        
        if token_embedder_type == "hashing":
                self.token_embedder = MultiHashingEmbedder((hidden_dim,\
                                                            slice_count= embedder_slice_count, bucket_count =  embedder_bucket_count))
        elif token_embedder_type == "None":
            self.token_embedder = nn.Embedding(input_dim, hidden_dim)
        else:
            RuntimeError(f"Invalid token embedder selected. Expected Hashing/None")
            
        if pos_embedder_type == "canine":
             self.position_embedder = PositionEmbedding(max_length, hidden_dim)
        elif pos_embedder_type == "None":
            self.position_embedder = PositionalEncoding(hidden_dim, dropout, max_length)
            
        else:
            RuntimeError(f"Invalid position embedder selected. Expected Canine/None")
        
        if not use_local_self_attention:
            self.layers = nn.ModuleList([EncoderLayer(hidden_dim,
                                                     num_attn_heads,
                                                     ff_dim,
                                                     dropout,
                                                     pos_embedder_type,
                                                     activation,
                                                     device)
                                        for _ in range(len(num_layers))])
        else:
            self.layers = nn.ModuleList([EncoderLayer(hidden_dim,
                                                     num_attn_heads,
                                                     ff_dim,
                                                     dropout,
                                                     window_size,
                                                     pos_embedder_type,
                                                    activation,
                                                     device,
                                                     use_local_self_attention)
                                        for _ in range(len(num_layers))])
            
        self.dropout = nn.Dropout(dropout)
        
        if self.pos_embedder_type == "None"
            self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        
        def forward(self, src, src_mask):
            
            #  src = [bs, src_len]
            #  src_mask = [bs, 1, 1, src_len]
            
            batch_size = src.shape[0]
            src_len = src.shape[1]
            
            if self.pos_embedder_type == "None":
                pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size,1).to(self.device)
                # pos = [bs, src_len]
                src = self.dropout(self.token_embedder(src)*self.scale) + self.position_embedder(pos)
            else:
                src = self.position_embedder(self.token_embedder(src))
                src = self.dropout(src)
                
            # src = [bs, src_len, hidden_dim]
            for L in self.layers:
                src = L(src, src_mask)
                
            # src = [bs, src_len, hidden_dim]
            
            return src
            
            
# Decoder
class Decoder(nn.Module):
    """
    Sequence-to-Sequence Decoder
    """
    
    def __init__(self,
                output_dim,
                hidden_dim,
                num_layers,
                num_attn_heads,
                ff_dim,
                dropout,
                device,
                max_length,
                embedder_slice_count : int = 0,
                embedder_bucket_count : int = 0,
                token_embedder_type  : str = "hashing",
                pos_embedder_type : str = "canine",
                use_local_self_attention = False,
                activation = "relu"
                ):
        super(Decoder,).__init__()
        
        self.device = device
        
        if activation not in ['relu', 'gelu']: RuntimeError(f"Invalid activation received. expects relu/gelu but got {activation}")
        else:
            self.activation = activation
            
        self.token_embedder_type = token_embedder_type
        self.pos_embedder_type = pos_embedder_type
        
        if token_embedder_type == "hashing":
                self.token_embedder = MultiHashingEmbedder((hidden_dim,\
                                                            slice_count= embedder_slice_count, bucket_count =  embedder_bucket_count))
        elif token_embedder_type == "None":
            self.token_embedder = nn.Embedding(input_dim, hidden_dim)
        else:
            RuntimeError(f"Invalid token embedder selected but expects Hashing/None")
            
        if pos_embedder_type == "canine":
             self.position_embedder = PositionEmbedding(max_length, hidden_dim)
        elif pos_embedder_type == "None":
            self.position_embedder = PositionalEncoding(hidden_dim, dropout, max_length)
        else:
            RuntimeError(f"Invalid position embedder selected but expects Canine/None")
        
        if not use_local_self_attention:
            self.layers = nn.ModuleList([DecoderLayer(hidden_dim,
                                                     num_attn_heads,
                                                     ff_dim,
                                                     dropout,
                                                     pos_embedder_type,
                                                     activation,
                                                     device)
                                        for _ in range(len(num_layers))])
        else:
            self.layers = nn.ModuleList([DecoderLayer(hidden_dim,
                                                     num_attn_heads,
                                                     ff_dim,
                                                     dropout,
                                                     window_size,
                                                     pos_embedder_type,
                                                    activation,
                                                     device,
                                                     use_local_self_attention)
                                        for _ in range(len(num_layers))])
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if self.pos_embedder_type == "None"
            self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)        
        
        
        def forward(self, tgt, enc_src, tgt_mask, src_mask):
            
            # tgt = [bs, tgt_len]
            # enc_src = [bs, src_len, hidden_size]
            # tgt_mask = [bs, 1, tgt_len, tgt_len]
            # src_mask = [bs, 1, 1, src_len]
            
            batch_size = tgt.shape[0]
            tgt_len    = tgt.shape[1]
            
            if self.pos_embedder_type == "None":
                pos  = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                
            # pos = [bs, tgt_len]
            
            if self.pos_embedder_type == "None":
                pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size,1).to(self.device)
                # pos = [bs, tgt_len]
                tgt = self.dropout(self.token_embedder(tgt)*self.scale) + self.position_embedder(pos)
            else:
                tgt = self.position_embedder(self.token_embedder(tgt))
                tgt = self.dropout(tgt) 
                
            # tgt = [ bs, tgt_len, hidden_dim ]
            
            for L in self.layers:
                tgt, attention = L(tgt, enc_src, tgt_mask, src_mask)
                
            # tgt = [bs, tgt_len, hidden_dim]
            # attention = [bs, num_attn_heads, tgt_len, src_len]
            
            attention = torch.mean(attention, dim = 1).unsqueeze(1).unsqueeze(2)
            
            return  attention
        
# seq2seq transducer
class TModel(nn.Module):
    """
    Sequence2Sequence model
    """
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 tgt_pad_idx,
                 device
                ):
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # src = [bs , src_len]
        src_mask =  (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # src_mask = [bs , 1, 1, src_len]
        
        return src_mask
    
    
    def make_tgt_mask(self, tgt):
        
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        
        tgt_len = tgt.shape[1]
        
        tgt_subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device = self.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_subsequent_mask
        
        return tgt_mask
    
    
    def forward(self, src, tgt):
        
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_src = self.encoder(src, src_mask)
        
        _ , attention = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        
        return attention
    