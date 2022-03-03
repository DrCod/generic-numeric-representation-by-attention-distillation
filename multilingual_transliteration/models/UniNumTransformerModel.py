import torch.nn as nn
import torch.nn.functional as F
import torch
from .local_self_attention import LocalSelfAttention
from .local_transformer_encoder_layer import LocalTransformerEncoderLayer
from .position_embedding import PositionalEncoding, PositionEmbedding
from .multi_hashing_embedder import MultiHashingEmbedder
from typing import Optional, Dict, Tuple
from types import SimpleNamespace
import json
import random
import math


global device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MHSA Layer
class MHSA(nn.Module):
    
    def __init__(self,
                hidden_dim,
                num_attn_heads,
                dropout,
                device):
        
        super(MHSA, self).__init__()
        
        assert hidden_dim % num_attn_heads == 0, "hidden_dim must be divisible by number of attention heads"
        
        self.hidden_dim = hidden_dim
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = hidden_dim // num_attn_heads
        
        self.Q_fc = nn.Linear(hidden_dim, hidden_dim)
        self.K_fc = nn.Linear(hidden_dim, hidden_dim)
        self.V_fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.attn_head_size])).to(device)
    
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]

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
        
        scores = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        
        # scores = [bs, num_attn_heads, query_len, key_len ]

#         if mask:

#             scores =  scores.masked_fill(mask == 0, -1e10)

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
        
        super(PositionwiseFeedforward, self).__init__()
        
        self.fc_1 = nn.Linear(hidden_dim, ff_dim)
        self.fc_2 = nn.Linear(ff_dim, hidden_dim)
        self.activation = activation
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # x = [bs, seq_len, hidden_dim]
        if self.activation == "relu":
            x = self.dropout(F.relu(self.fc_1(x)))
        elif self.activation == "gelu":
            x = self.dropout(F.gelu(self.fc_1(x)))
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
#                  pos_embedder_type,
                 activation,
                 device,
                 window_size = 0,
                 use_local_self_attention = None
                ):
        super(EncoderLayer, self).__init__()
        
#         if window_size % 2 != 0:
#             raise ValueError('Window size must be an even number so it can be split evenly across previous and' 'following tokens.')
            
        self.self_attention = LocalSelfAttention(hidden_dim, num_attn_heads, window_size,dropout,auto_pad =False) if use_local_self_attention else MHSA(hidden_dim, num_attn_heads, dropout, device)
        
        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.ff_ln   = nn.LayerNorm(hidden_dim)
        self.positionwise_ff = PositionwiseFeedforward(hidden_dim, ff_dim, dropout, activation)
        
        self.dropout = nn.Dropout(dropout)
        self.use_local_self_attention = use_local_self_attention
#         self.pos_embedder_type = pos_embedder_type
        
        
    def forward(self, src, src_mask):

        # src = [bs, src_len, hidden_dim]
        # src_mask = [ bs, 1, 1, src_len]
        
        if not self.use_local_self_attention:
            _src, _ = self.self_attention(src, src, src, src_mask)
        else:
            _src =  self.self_attention(src, src_mask)

        # dropout, residual connection and layer norm
        src = self.attn_ln(src + self.dropout(_src))

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
                use_local_self_attention = None,
                embedder_slice_count : int = 0,
                embedder_bucket_count : int = 0,
                window_size : int = 0,
                activation = "relu"
                ):
        
        super(Encoder, self).__init__()

        self.device = device

        if activation not in ['relu', 'gelu']: RuntimeError(f"Invalid activation. Expects relu/gelu but got {activation}")
        else: self.activation = activation

#         self.token_embedder = token_embedder
#         self.pos_embedder = pos_embedder

        #         if pos_embedder_type == "normal":
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        
        #         if token_embedder_type == "hashing":
        #             self.token_embedder = MultiHashingEmbedder(hidden_dim, slice_count= embedder_slice_count, bucket_count =  embedder_bucket_count)
        #         elif token_embedder_type == "normal":
        #             self.token_embedder = nn.Embedding(input_dim, hidden_dim)
        #         else:
        #             RuntimeError(f"Invalid token embedder selected. Expected hashing/None")

        #         if pos_embedder_type == "canine":
        #             self.position_embedder = PositionEmbedding(max_length, hidden_dim)
        #         elif pos_embedder_type == "attn_paper":
        #             self.position_embedder = PositionalEncoding(hidden_dim, dropout, max_length)
        #         else:
        #             RuntimeError(f"Invalid position embedder selected. Expected Canine/None")

        if not use_local_self_attention:
            self.layers = nn.ModuleList([EncoderLayer(hidden_dim,
                                                     num_attn_heads,
                                                     ff_dim,
                                                     dropout,
                                                     activation,
                                                     device)
                                        for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([EncoderLayer(hidden_dim,
                                                     num_attn_heads,
                                                     ff_dim,
                                                     dropout,
                                                     window_size,
                                                     activation,
                                                     device,
                                                     use_local_self_attention)
                                        for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
            
        
    def forward(self, src, src_mask):

        #  src = [bs, src_len]
        #  src_mask = [bs, 1, 1, src_len]

        batch_size = src.shape[0]
        src_len    = src.shape[1]

        # src = [bs, src_len, hidden_dim]
        for L in self.layers:
            src = L(src, src_mask)

        # src = [bs, src_len, hidden_dim]

        return src

        
#Decoder Layer

class DecoderLayer(nn.Module):
    
    """
    Decoder Layer
    """
    
    def __init__(self,
                 hidden_dim,
                 num_attn_heads,
                 ff_dim,
                 dropout,
                 activation,
                 device,
                 window_size = 0,
                 use_local_self_attention = None
                ):
        super(DecoderLayer, self).__init__()

#         if window_size % 2 != 0:
#             raise ValueError('Window size must be an even number so it can be split evenly across previous and' 'following tokens.')
            
        # encoder_attention
        self.encoder_self_attention = LocalSelfAttention(hidden_dim, num_attn_heads, window_size, dropout,auto_pad =False) if use_local_self_attention else MHSA(hidden_dim, num_attn_heads, dropout, device)
        
        self.decoder_self_attention = MHSA(hidden_dim, num_attn_heads, dropout, device)
        
        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.dec_attn_ln =  nn.LayerNorm(hidden_dim)
        self.positionwise_ff = PositionwiseFeedforward(hidden_dim, ff_dim, dropout, activation)
        self.hidden_dim = hidden_dim
        self.num_attn_heads = num_attn_heads
        self.ff_dim = ff_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.device = device
        self.window_size = window_size
        self.use_local_self_attention = use_local_self_attention
        
                
    def forward(self, tgt, enc_src=None, tgt_mask =None, src_mask =None):
        
        if tgt_mask is not None:
            _tgt, attention = self.decoder_self_attention(tgt, tgt, tgt, tgt_mask)
        else:
            _tgt, attention = self.decoder_self_attention(tgt, tgt, tgt)            
        
        tgt  = self.attn_ln(tgt + self.dropout(_tgt))
        
        if enc_src is not None:
            _tgt, attention = self.decoder_self_attention(tgt, enc_src, enc_src, src_mask)

        tgt = self.dec_attn_ln(_tgt + self.dropout(_tgt))
        
        _tgt = self.positionwise_ff(tgt)
        
        tgt = tgt + self.attn_ln(tgt + self.dropout(_tgt))
        
        return tgt, attention

    
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
                use_local_self_attention = False,
                activation = "relu"
                ):
        super(Decoder, self).__init__()
        
        self.device = device
        
        if activation not in ['relu', 'gelu']: 
            RuntimeError(f"Invalid activation received. expects relu/gelu but got {activation}")
        else: 
            self.activation = activation
            
        #         self.token_embedder = token_embedder
        #         self.pos_embedder   = pos_embedder
        
        #         if token_embedder_type == "hashing":
        #             self.token_embedder = MultiHashingEmbedder(hidden_dim,\
        #                                                             slice_count= embedder_slice_count, bucket_count =  embedder_bucket_count)
        #         elif token_embedder_type == "normal":
        #             self.token_embedder = nn.Embedding(output_dim, hidden_dim)
        #         else:
        #             RuntimeError(f"Invalid token embedder selected but expects Hashing/None")

        #         if pos_embedder_type == "canine":
        #              self.position_embedder = PositionEmbedding(max_length, hidden_dim)
        #         elif pos_embedder_type == "attn_paper":
        #             self.position_embedder = PositionalEncoding(hidden_dim, dropout, max_length)
        #         else:
        #             RuntimeError(f"Invalid position embedder selected but expects Canine/None")
        
        if not use_local_self_attention:
            self.layers = nn.ModuleList([DecoderLayer(hidden_dim,
                                                     num_attn_heads,
                                                     ff_dim,
                                                     dropout,
                                                     activation,
                                                     device)
                                        for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([DecoderLayer(hidden_dim,
                                                     num_attn_heads,
                                                     ff_dim,
                                                     dropout,
                                                     window_size,
                                                     activation,
                                                     device,
                                                     use_local_self_attention)
                                        for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
#         if self.pos_embedder_type == "attn_paper":
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)        
        

    def forward(self, tgt, enc_src, tgt_mask, src_mask):

        # tgt = [bs, tgt_len]
        # enc_src = [bs, src_len, hidden_size]
        # tgt_mask = [bs, 1, tgt_len, tgt_len]
        # src_mask = [bs, 1, 1, src_len]

        batch_size = tgt.shape[0]
        tgt_len    = tgt.shape[1]

        #         if self.pos_embedder_type == "attn_paper":
        #         pos  = torch.arange(0, tgt_len).long().unsqueeze(0).repeat(batch_size, 1).to(self.device)

                # pos = [bs, tgt_len]

        #         if self.pos_embedder_type == "attn_paper":
        #         pos = torch.arange(0, tgt_len).long().unsqueeze(0).repeat(batch_size,1).to(self.device)
                # pos = [bs, tgt_len]
        #         tgt = self.dropout(self.token_embedder(tgt)*self.scale) + self.position_embedder(pos)
        #         else:
        #             tgt = self.position_embedder(self.token_embedder(tgt))
        #             tgt = self.dropout(tgt) 

        # tgt = [ bs, tgt_len, hidden_dim ]

        for L in self.layers:
            tgt, attention = L(tgt, enc_src, tgt_mask, src_mask)

        # tgt = [bs, tgt_len, hidden_dim]
        # attention = [bs, num_attn_heads, tgt_len, src_len]

        x = F.adaptive_avg_pool2d(attention, (1, 1))

        return  tgt, x
        

class UniversalNumericalTransformer(nn.Module):
    """Universal Numerical Transformer Model
    A transformer model that learn to represent universal numerical text sequences multilingually using local transformer encoder 
    and multihashing embedding utilities from CANINE paper
    
    Args:
        intoken, outtoken (int): Number of tokens in both input and output text
        hidden (int): Dimension of the model (d_model)
        dropout: dropout
        attetnion_heads: Number of attention heads
        pad_token: The padding token

    """
    # Adapted some Defaults from CANINE
    def __init__(self,
                intoken  : int = 0,
                outtoken : int = 0,
                downsampling_rate : int = 4,
                upsampling_kernel_size : int = 4,
                embedder_slice_count : int = 8,
                embedder_bucket_count : int = 16000,
                hidden_size : int = 768,
                local_attention_window : int = 128,
                deep_transformer_stack : Optional[nn.Module] = None,
                deep_transformer_requires_transpose : bool = True,
                attention_heads : int = 4,
                transformer_ff_size : int = 3072,
                dropout : float = 0.1,
                activation : str = 'gelu',
                pad_token : int = 0,
                max_length : int = 32,
                deep_transformer_stack_layers : Optional[int] = None,
                token_embedding_type : str = 'hashing',
                position_embedding_type : str = 'canine',
                use_local_transformer  = None
                ):
        
        super(UniversalNumericalTransformer, self).__init__()
        
        # Init all hyper-parameters here
        self.intoken = intoken
        self.outtoken = outtoken
        self.downsampling_rate = downsampling_rate
        self.upsampling_kernel_size = upsampling_kernel_size
        self.embedder_slice_count = embedder_slice_count
        self.embedder_bucket_count = embedder_bucket_count
        self.hidden_size = hidden_size
        self.local_attention_window = local_attention_window
        self.deep_transformer_stack = deep_transformer_stack
        self.deep_transformer_requires_transpose = deep_transformer_requires_transpose
        self.attention_heads = attention_heads
        self.transformer_ff_size = transformer_ff_size
        self.dropout  = dropout
        self.activation = activation
        self.pad_token = pad_token
        self.max_length  = max_length
        self.deep_transformer_stack_layers = deep_transformer_stack_layers
        self.token_embedding_type  = token_embedding_type
        self.position_embedding_type = position_embedding_type
        self.use_local_transformer = use_local_transformer
        

#         assert max_length % downsampling_rate == 0, f"max length must be divisible by downsampling rate, but got {max_length} and {downsampling_rate} respectively"
        
        activations = {
                       'gelu' : nn.GELU,
                       'relu' : nn.ReLU
                      }
        tok_embeddings = {'hashing' : MultiHashingEmbedder,
                          'normal'  : nn.Embedding
                         }
        
        pos_embeddings = {'canine' : PositionEmbedding,
                          'attn_paper' : PositionalEncoding
                         }
        
        if activation not in activations:
            RuntimeError(f"activation must be  in {activations.keys()} but is {activation}")
        else:
            self.activation_fn = activations[activation]()
            
            
        if token_embedding_type == "hashing":
            self.encoder_token_embedder = tok_embeddings[token_embedding_type](hidden_size, slice_count = embedder_slice_count, 
                                                                       bucket_count = embedder_bucket_count)
            self.decoder_token_embedder = tok_embeddings[token_embedding_type](hidden_size, slice_count = embedder_slice_count, 
                                                                       bucket_count = embedder_bucket_count)
            
        elif token_embedding_type == "normal":
            self.encoder_token_embedder = tok_embeddings[token_embedding_type](self.intoken, hidden_size)
            self.decoder_token_embedder = tok_embeddings[token_embedding_type](self.outtoken, hidden_size)
        else:
            RuntimeError(f"token embedding type must be in {tok_embeddings.keys()} but is {token_embedding_type}")
                     
        if position_embedding_type == "canine":
            self.position_embedder = pos_embeddings[position_embedding_type](max_length, hidden_size)
            
        elif position_embedding_type == "attn_paper" :
            self.position_embedder = pos_embeddings[position_embedding_type](hidden_size, dropout)
        else:
            RuntimeError(f"position embedding type must be in {pos_embeddings.keys()} but is {position_embedding_type}")
                   
        self.dropout = nn.Dropout(p = dropout)
        
        self.embedder_ln = nn.LayerNorm(hidden_size)

        # "Single Local Transformer"
        # note the CANINE paper says "local transformer", but it means "local transformer encoder" just like BERT
        
        if self.use_local_transformer:
            
            self.encoder = LocalTransformerEncoderLayer(hidden_size, attention_heads, window_size = local_attention_window, dropout=dropout,
                                                                  activation= activation,
                                                                  dim_feedforward=transformer_ff_size)


            #             self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size,
            #                                                                              attention_heads,
            #                                                                              dim_feedforward=transformer_ff_size,
            #                                                                              dropout=dropout,
            #                                                                              activation=activation),
            #                                                                 num_layers= 1)

            # "Downsample (Strided Convolution) "
            self.downsample_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=downsampling_rate,
                                               stride=downsampling_rate)

            self.downsample_attention_pool = torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)
            self.downsample_ln = torch.nn.LayerNorm(hidden_size)
            self.cls_linear = torch.nn.Linear(hidden_size, hidden_size)

            # "Deep Transformer Stack"
            if deep_transformer_stack is not None:
                if deep_transformer_stack_layers is not None:
                    raise RuntimeError('deep_transformer_stack_layers and deep_transformer_stack both provided - please '
                                       'provide only one.')
                    # TODO: perform some kind of basic verification that this is actually a torch module that can be used
                    # in place of the default deep transformer stack
                self.deep_transformer = deep_transformer_stack
                
            else:
                layers = deep_transformer_stack_layers if deep_transformer_stack_layers is not None else 2

            self.deep_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size,
                                                                                 attention_heads,
                                                                                 dim_feedforward=transformer_ff_size,
                                                                                 dropout=dropout,
                                                                                 activation=activation),
                                                num_layers=layers)

            self.deep_transformer_requires_transpose = True

            # "Conv + Single Transformer"
            self.upsample_conv = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=upsampling_kernel_size, stride=1)
            self.upsample_ln = nn.LayerNorm(hidden_size)
            self.final_transformer =  nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, attention_heads,
                                                                                     dim_feedforward=transformer_ff_size,
                                                                                     dropout=dropout,
                                                                                     activation=activation),
                                                    1)
        
            decoder_layers = nn.TransformerDecoderLayer(hidden_size, attention_heads,
                                                         dim_feedforward=transformer_ff_size,
                                                         dropout=dropout,
                                                         activation=activation)

            self.decoder = nn.TransformerDecoder(decoder_layers, 1)
            
        else:
            self.encoder = Encoder(
                        input_dim = intoken,
                        hidden_dim = hidden_size,
                        num_layers = 1,
                        num_attn_heads = attention_heads,
                        ff_dim = transformer_ff_size,
                        dropout = dropout,
                        device =device,
                        max_length = max_length,
                        embedder_slice_count = embedder_slice_count,
                        embedder_bucket_count  = embedder_bucket_count,
#                         token_embedder  = self.encoder_token_embedder,
#                         pos_embedder = self.position_embedder,
                        use_local_self_attention = None,
                        activation = activation
                        )

            self.decoder = Decoder(
                        output_dim = outtoken,
                        hidden_dim = hidden_size,
                        num_layers = 1,
                        num_attn_heads = attention_heads,
                        ff_dim = transformer_ff_size,
                        dropout = dropout,
                        device =device,
                        max_length = max_length,
                        embedder_slice_count = embedder_slice_count,
                        embedder_bucket_count  = embedder_bucket_count,
#                         token_embedder  = self.decoder_token_embedder,
#                         pos_embedder = self.position_embedder,
                        use_local_self_attention = None,
                        activation = activation
                        )

        
        # CLS Token
        self.cls_linear_final = nn.Linear(hidden_size, hidden_size)
        self.cls_activation = nn.Tanh()

        # Classifier Head
        self.fc = nn.Linear(hidden_size , outtoken)
        
        
    # "Upsampling"
    def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: int):

        repeated_molecules = molecules.repeat_interleave(self.downsampling_rate, axis=1)
        remainder_length = char_seq_length % self.downsampling_rate

        # as the canine implementation does, we repeat the last molecule extra times to get to a multiple of 4
        last_molecule = molecules[:, -1:, :]
        last_molecule_repeated = last_molecule.repeat_interleave(remainder_length, axis=1)

        return torch.cat((repeated_molecules, last_molecule_repeated), dim=1)
    
    
    def get_enc_token_embedder(self):
        return self.encoder_token_embedder
    
    def get_dec_token_embedder(self):
        return self.decoder_token_embedder
    
    def get_position_encoder(self):
        return self.position_embedder
    
    def get_transformer_encoder(self):
        return self.encoder
    
    def get_transformer_decoder(self):
        return self.decoder
    
    def get_fc(self):
        return self.fc
    
    def forward(self, input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                tgt_input_ids : torch.Tensor,
                tgt_attention_mask : torch.Tensor
               ):
        
        trg_mask = self._generate_square_subsequent_mask(len(tgt_input_ids)).to(tgt_input_ids.device)
                
        char_embeddings = self.position_embedder(self.encoder_token_embedder(input_ids))
        
        tgt_char_embeddings = self.position_embedder(self.decoder_token_embedder(tgt_input_ids))
        
        
        if self.use_local_transformer:
    
            contextualized_chars = self.encoder(char_embeddings, attention_mask)

            sampleable_characters = contextualized_chars.transpose(1, 2).contiguous()
            sampleable_mask = attention_mask.float()

            molecules = self.downsample_conv(sampleable_characters).transpose(1, 2)  # h_down
            molecules = self.downsample_ln(self.activation_fn(molecules))
            molecules = torch.cat((contextualized_chars, molecules), dim=1)        

            # unlike CANINE we don't assume a fixed size and truncate, so we have to add to the attention mask for the
            # CLS slot. squeezing and unsqueezing is a fix for https://github.com/pytorch/pytorch/issues/51954
            downsampled_attention_mask = self.downsample_attention_pool(sampleable_mask.unsqueeze(0)).squeeze(0)
            molecule_attention_mask = torch.nn.functional.pad(downsampled_attention_mask.bool(), (1, 0), value=True)

            # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L343
            if self.deep_transformer_requires_transpose:
                # TODO: if we switch out the deep transformer to something that calls its attention mask
                # anything other than "src_key_padding_mask" this will break
                contextualized_molecules = self.deep_transformer(molecules.transpose(0, 1),
                                                            src_key_padding_mask=molecule_attention_mask).transpose(0, 1)  # h`_down
            else:
                contextualized_molecules = self.deep_transformer(molecules, src_key_padding_mask=molecule_attention_mask)

            # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L371
            molecules_without_cls = contextualized_molecules[:, 1:, :]  # remove CLS to avoid upsampling it
            repeated_molecules = self._repeat_molecules(molecules_without_cls, contextualized_chars.shape[1])

            # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L468
            concatenated = torch.cat((contextualized_chars, repeated_molecules), dim=2)
            concatenated = self._pad_for_convolution_to_same_length(concatenated, self.upsample_conv)
            upsampled_embeddings = self.activation_fn(self.upsample_conv(concatenated.transpose(1, 2).contiguous()).
                                                   transpose(1, 2))
            upsampled_embeddings = self.dropout(self.upsample_ln(upsampled_embeddings))  # h_up

            # https://github.com/google-research/language/blob/master/language/canine/modeling.py#L551
            contextualized_cls = contextualized_molecules[:, 0:1, :]
            final_cls = self.cls_activation(self.cls_linear_final(contextualized_cls))

            # confusingly, key_padding_mask does for the pytorch transformer what attention_mask does for the
            # local attention implementation (and huggingface/allennlp)
            # also, we drop the first embedding (CLS token) because we're going to use final_cls anyway
            final_embeddings = self.final_transformer(upsampled_embeddings[:, 1:, :].transpose(0, 1),
                                                      src_key_padding_mask=attention_mask[:, 1:]).transpose(0, 1)
            final_cls_embeddings = torch.cat((final_cls, final_embeddings), dim=1)  # replace CLS embedding

        else:
            
            final_cls_embeddings = self.encoder(char_embeddings, attention_mask)

            # Initialize the decoder with the final encoder embeddings
            #         decoder_final_states, attn_probas = self.decoder(tgt = tgt_char_embeddings, 
            #                                    memory = final_cls_embeddings, tgt_mask = trg_mask, memory_mask = None,
            #                                   tgt_key_padding_mask = tgt_attention_mask, memory_key_padding_mask = attention_mask
            #                                    )

        decoder_final_states, attn_probas = self.decoder(tgt = tgt_char_embeddings, 
                           enc_src = None,
                                                         tgt_mask = trg_mask, src_mask = attention_mask
                           )
                        
        output = self.fc(decoder_final_states)
            
        return {
            'embeddings': final_cls_embeddings,
            'attn_probas': attn_probas,
            'logits': output
        }
    
    """Triangular mask for attention: https://arxiv.org/pdf/1706.03762.pdf
    """
    def _generate_square_subsequent_mask(self, sz, sz1=None):
        
        if sz1 == None:
            mask = torch.triu(torch.ones(sz, sz), 1)
        else:
            mask = torch.triu(torch.ones(sz, sz1), 1)
            
        return mask.masked_fill(mask==1, float('-inf'))

    def _pad_to_avoid_missed_characters(self, char_embeddings: torch.Tensor) -> torch.Tensor:
        if char_embeddings.shape[1] % self.downsampling_rate == 0:
            return char_embeddings
        else:
            target_length = math.ceil(char_embeddings.shape[1] / self.downsampling_rate)\
                            * self.downsampling_rate
            total_padding = target_length - char_embeddings.shape[1]
            lhs_padding = math.floor(total_padding / 2)
            rhs_padding = math.ceil(total_padding / 2)
            return torch.nn.functional.pad(char_embeddings, (0, 0, lhs_padding, rhs_padding))

    def _pad_for_convolution_to_same_length(self, hidden_state: torch.Tensor,
                                            convolution: torch.nn.Conv1d) -> torch.Tensor:
        
        # we have to manually pad, see: https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/2
        # so we solve for total padding from the formula for output length
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # hidden state has shape [batch_size, sequence_length, embedding_size]
        
        l = hidden_state.shape[1]
        s = convolution.stride[0]
        d = convolution.dilation[0]
        k = convolution.kernel_size[0]

        total_padding = l * s - l + d * k - d + 1 - s
        lhs_padding = math.floor(total_padding / 2)
        rhs_padding = math.ceil(total_padding / 2)

        return torch.nn.functional.pad(hidden_state, (0, 0, lhs_padding, rhs_padding))