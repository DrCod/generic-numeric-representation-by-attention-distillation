import os
import sys
import torch
import torch.nn as nn
from .multi_hash_embedding import MultiHashingEmbedder
from .position_embedders import PositionEmbedding, PositionalEncoding
import torch.nn.functional as F
from .local_self_attention import LocalSelfAttention
from .seq2seq_T import *

class UniversalRepresentationModel(nn.Module):
    
    def __init__(self,
                 hparams,
                 vocab_size,
                 dim_attn_p,
                ):
        
        super(UniversalRepresentationModel, self).__init__()

        self.hparams = hparams

        n_classes = hparams.n_classes

        Ci = 1 # number of input channels
        Co = hparams.channel_out
        Kernel_sizes = hparams.kernel_sizes # List of kernels

        if hparams.token_embedder_type == "hashing":
            self.token_embedder = MultiHashingEmbedder(hparams.hidden_dim, slice_count = hparams.embedder_slice_count, 
                                                                           bucket_count =  hparams.embedder_bucket_count)
        elif hparams.token_embedder_type == "None":
            self.token_embedder = nn.Embedding(vocab_size, hparams.hidden_size)
        else:
            RuntimeError(f"Invalid token embedder selected . Token embedder must be in hashing/None")

        self.convs = nn.ModuleList([nn.Conv1D(Ci, Co, (K, hparams.hidden_size)) for K in Kernel_sizes])
        self.dropout = nn.Dropout(p = hparams.dropout)

        # final prediction layers
        self.num_pred_layer = nn.Linear(hparams.hidden_size, n_classes)
        self.attn_pred_layer = nn.Linear(hparams.hidden_size, dim_attn_p)


        def forward(self, x):

            char_emb = self.token_embedder(x)
            char_emb = char_emb.unsqueeze(1)

            x = [F.relu(conv(char_emb)).squeeze(3) for conv in self.convs]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

            final_embeddings = torch.cat(x, 1)

            num_logits = self.num_pred_layer(final_embeddings)
            attn_logits = self.attn_pred_layer(final_embeddings)

            return {'generic_embeddings': final_embeddings,
                   'num_logits': num_logits,
                    'attn_logits': attn_logits
                   }
        
        
