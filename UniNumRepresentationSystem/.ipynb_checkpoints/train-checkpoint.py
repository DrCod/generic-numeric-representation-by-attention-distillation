import os
import sys
import json
import random
import math
import torch
import time
import torch.nn as nn
from models.multi_hash_embedding import MultiHashingEmbedder
from models.position_embedders import PositionEmbedding, PositionalEncoding
import torch.nn.functional as F
from models.local_self_attention import LocalSelfAttention
from models.seq2seq_T import TModel, Encoder, Decoder
from models.U import UniversalRepresentationModel
import argparse
import logging
from torch.utils.data import DataLoader
from dataloaders.dataloader import NumericalDataset,NumericalDatasetCollator
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import *


def run_epoch(model,t_model, epoch, iterator, optimizer, criterion_dict, device,  in_token_to_int, out_token_to_int):
    """Perform one training epoch
    """

    model.train()

    pbar = tqdm(iter(iterator), leave=True, total=len(iterator))

    start_iter = 0
    total_loss = 0
    train_acc = []
    start_time = time.time()

    sys.stdout.flush()

    for i , (data) in enumerate(pbar, start = start_iter):

        X, y, aux_data = data
        
        X_tokenized_padded = list(map(lambda seq : tokenize_in(seq, in_token_to_int), X))
        X = torch.Tensor(X_tokenized_padded).long()
        
        # move tensors to device
        X = X.to(device)
        y = y.to(device)
        
        # unpack the auxillary data
        src_list , tgt_list = aux_data
        # tokenize and pad to max_length        
        src_tok = list(map(lambda  x : tokenize_in(x, in_token_to_int), src_list))
        tgt_tok = list(map(lambda x : tokenize_out(x, out_token_to_int), tgt_list))
        
        src = torch.Tensor(src_tok).long().to(device)
        tgt = torch.Tensor(tgt_tok).long().to(device)
        
        # Compute lexical and semantic similarity attention probability
        # distribution over source and target sequences
        y_attn = t_model(src, tgt)
        
        output_dict =  model(X)
        
        y_num_logits  = output_dict['num_logits']
        y_attn_logits = output_dict['attn_logits']
        learned_repr  = output_dict['generic_embeddings']

        optimizer.zero_grad()
        #L1 : CrossEntropy loss
        L1 = criterion_dict['L1'](y_num_logits, y)
        #l2: KLDivLoss
        L2 = criterion_dict['L2'](y_attn, y_attn_logits)
        
        L = L1.item() + L2.item()
        
        total_loss += L.item()

        L.backward()

        acc = calculate_accuracy(y_num_logits, y)
        train_acc.append(acc)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    logging.info("(Epoch {}) LOSS:{:.4f} ACC:{:.3f}".format(
            (epoch+1), total_loss / len(iterator), np.mean(train_acc)))

    return total_loss / len(iterator), model

def run_validation(model, t_model ,epoch, iterator, criterion_dict,  in_token_to_int, out_token_to_int):
    """Get validation loss
    """        
    model.eval()

    pbar = tqdm(iter(iterator), leave=True, total=len(iterator))

    val_acc = []
    start_iter = 0
    total_loss = 0
    start_time = time.time()

    sys.stdout.flush()

    for i, (data) in enumerate(pbar , start = start_iter):

        X, y, aux_data = data
        
        X_tokenized_padded = list(map(lambda seq : tokenize_in(seq, in_token_to_int), X))
        X = torch.Tensor(X_tokenized_padded).long()
        
        # move tensors to device
        X = X.to(device)
        y = y.to(device)
        
        # unpack the auxillary data
        src_list , tgt_list = aux_data
        # tokenize and pad to max_length        
        src_tok = list(map(lambda  x : tokenize_in(x, in_token_to_int), src_list))
        tgt_tok = list(map(lambda x : tokenize_out(x, out_token_to_int), tgt_list))
        
        src = torch.Tensor(src_tok).long().to(device)
        tgt = torch.Tensor(tgt_tok).long().to(device)
        
        # Compute lexical and semantic similarity attention probability
        # distribution over source and target sequences
        y_attn = t_model(src, tgt)
        
        output_dict =  model(X)
        
        y_num_logits  = output_dict['num_logits']
        y_attn_logits = output_dict['attn_logits']
        learned_repr  = output_dict['generic_embeddings']

        acc = calculate_accuracy(y_num_logits, y)
        val_acc.append(acc)

    logging.info("(Epoch {}) LOSS:{:.4f} ACC:{:3f}".format(
        (epoch+1), total_loss / len(iterator), np.mean(val_acc)))

    return total_loss / len(iterator)


def main(args):

    pad_token = 0
    eos_token = 2
    sos_token = 1


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"DEVICE: {device}")

    df = pd.read_csv(f"../data/{args.auxillary_data_csv}")[['source', 'target']]
    
    # preprocessing
    df['source'] = df['source'].apply(lambda x : preprocess_text(x))
    df['target'] = df['target'].apply(lambda x : preprocess_text(x))
    
    #Create the dictionary that maps each letter to it's corresponding embedding token
    #Input has one special token for padding
    in_tokens =  set(" ".join(df['source'].values.tolist()))
    in_token_to_int = {token: (i+1) for i,token in enumerate(sorted(in_tokens))}
    
    in_max = max(list(map(lambda x : len(x), df['source'].values.tolist())))
    out_max = max(list(map(lambda x : len(x), df['target'].values.tolist()))) + 2 # account for <eos> and <sos> tokens
                 

    in_token_to_int[0] = "<pad>"

    #Out put has three special tokens, <eos> <sos> and <pad>
    out_tokens = set(" ".join(df['target'].values.tolist()))

    out_token_to_int = {token: (i+3) for i,token in enumerate(sorted(out_tokens))}

    out_token_to_int["<pad>"] = pad_token
    out_token_to_int["<sos>"] = sos_token
    out_token_to_int["<eos>"] = eos_token

    out_int_to_token = {out_token_to_int[t]:t for t in out_token_to_int}
                
    encoder = Encoder(
                input_dim = len(in_token_to_int),
                hidden_dim = args.hidden_dim,
                num_layers = args.n_layers,
                num_attn_heads = args.n_heads,
                ff_dim = args.feedforward_dim,
                dropout = args.dropout,
                device =device,
                max_length = args.max_length,
                embedder_slice_count = args.embedder_slice_count,
                embedder_bucket_count  = args.embedder_bucket_count,
                token_embedder_type  = args.token_embedder_type,
                pos_embedder_type = args.pos_embedder_type,
                use_local_self_attention = False,
                activation = args.activation
                )
    
    decoder = Decoder(
                output_dim = len(out_token_to_int),
                hidden_dim = args.hidden_dim,
                num_layers = args.n_layers,
                num_attn_heads = args.n_heads,
                ff_dim = args.feedforward_dim,
                dropout = args.dropout,
                device =device,
                max_length = args.max_length,
                embedder_slice_count = args.embedder_slice_count,
                embedder_bucket_count  = args.embedder_bucket_count,
                token_embedder_type  =args.token_embedder_type,
                pos_embedder_type = args.pos_embedder_type,
                use_local_self_attention = args.use_local_transformer,
                activation = args.activation
                )
            
    seq2seq = TModel(
                 encoder,
                 decoder,
                 pad_token,
                 device
                )
    
    u_model = UniversalRepresentationModel(
        args,
        len(in_token_to_int),
        args.max_length
    )
    
    dataset = NumericalDataset(args.train_csv. args.auxillary_data_csv, args.k)
    
    ntrain = len(dataset)
    train_split = int(ntrain * (1 - args.val_split))
    val_split = ntrain - train_split
    
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_split, val_split])
    
    train_dataloader = DataLoader(train_set, 
                        batch_size= args.batch_size // self.args.gradient_accumulation_steps , collate_fn= NumericalDatasetCollator)
    valid_dataloader = DataLoader(valid_set,
                                 batch_size = self.args.batch_size // self.args.gradient_accumulation_steps , collate_fn = NumericalDatasetCollator)
    
    
    
    logging.info("*"*30)
    logging.info("Training Universal Represeantaion System ...")
    logging.info(f"{args.epochs} epochs, {args.batch_size} batch_size")
    logging.info("*"*30)

    logging.info("Create dataloaders")

    if args.use_label_smoothing:
        ce_criterion = LabelSmoothing(size= 10, padding_idx= out_token_to_int["<pad>"], smoothing= args.smoothing_param)
    else:
        ce_criterion = nn.CrossEntropyLoss(ignore_index= out_token_to_int["<pad>"])
        
    kl_criterion = nn.KLDivLoss()
    
    criterion_dict = {'L1': ce_criterion,
                      'L2': kl_criterion
                     }

    optimizer = optim.Adam(u_model.parameters(), lr = args.learning_rate)

    logging.info("Training...")

    min_loss = 99
    bad_epoch = 0
    best_epoch = 0
    #Change model size
    for i in range(args.epochs):

        loss, trained_model = run_epoch(u_model, seq2seq, i, train_dataloader, optimizer, criterion_dict, in_token_to_int, out_token_to_int)

        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps


        loss_val = run_validation(trained_model, seq2seq, i, valid_dataloader, criterion_dict,  in_token_to_int, out_token_to_int)

        if loss_val < min_loss:
            min_loss = loss_val
            best_epoch = i

            # save best model
            save_model(trained_model, seq2seq, i, optimizer, in_token_to_int,
                       out_token_to_int, out_int_to_token, args, 
                       f"saved_models/generic_repr_best_val_loss_{min_loss}_epoch_{best_epoch}.pth")

            logging.info("New best loss %f" % (min_loss))

        elif loss_val > min_loss:
            bad_epoch += 1
            if bad_epoch >= args.patience:
                break

        logging.info("EPOCH %d -- %f -- Val Loss: %f" % (i, loss, loss_val))

    logging.info("="*20)
    logging.info("Training done, best loss %f" % (min_loss))
    logging.info("="*20)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description ="Train Generic Numerical Representation System")
    parser.add_argument("--epochs", help="number of training epochs", default=100, type=int)
    parser.add_argument("--batch_size", help="Batch size. Default: 32", default= 32, type=int)
    parser.add_argument("--d_model", help="Transformer model dimension. Default 512", default=512, type=int)
    parser.add_argument("--train_csv",type = str, default = None, help ="train csv file")
    parser.add_argument("--auxillary_data_csv", type = str, default = None, help ="auxillary train csv file")
    parser.add_argument("--k", type = int, default = 1, help ="sampling parameter")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--learning_rate",default = 1e-3, type = float, help = "learning rate")
    parser.add_argument("--token_embedder_type", type=str, default = "hashing", choices =['hashing', 'None'], help = "token embedding  technique")
    parser.add_argument("--pos_embedder_type", type=str, default = "canine", choices =["canine", 'None'], help="position encoding technique")
    parser.add_argument("--dropout", type= float, default=0.0 , help = "dropout rate")
    parser.add_argument("--n_heads",default = 4, type = int, help ="number of attention heads" )
    parser.add_argument("--use_local_transformer" ,action="store_true", help = "use local transformer encoder")
    parser.add_argument("--activation", type = str, default ="relu", choices = ["gelu", "relu"], help ="activation function")
    parser.add_argument("--feedforward_dim", type = int , default = 2048, help = "transformer feed forward dimension")
    parser.add_argument("--max_length", type = int, default = 32, help = "maximum length of input sequence")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--patience", type = int, default = 5, help = "patience parameter to modulate early stopping")
    parser.add_argument("--use_label_smoothing", action="store_true", default =False, help = "toggle between using a using label smoothing or not")
    parser.add_argument("--smoothing_param", type = float, default = 0.0, help = "label smoothing parameter")
    parser.add_argument("--val_split", default = 0.1, type = float, help ="validation split")
    parser.add_argument("--embedder_slice_count",  default = 8, type = int, help = "embedder slice count")
    parser.add_argument("--embedder_bucket_count", default = 16000, type = int, help = "embedder bucket count")
    parser.add_argument("--hidden_dim", default = 768, type = int, help = "embedding dimension" )
    parser.add_argument("--n_layers", default = 1, type = int, help = "number of encoder/decoder layers")
    parser.add_argument("--n_classes", default = 9, type = int , help = "number of classes")
    parser.add_argument("--kernel_sizes", type = str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument("--channel_out", type = int, default = 100, help = "number of each type of kernels" )
    
    # parser.add_argument("--lang", type=str, choices = ["ita", "ara","deu","afr","abk","aka","yor","jpn","spa","fra"], help ="language type")

    args = parser.parse_args()


    main(args)
