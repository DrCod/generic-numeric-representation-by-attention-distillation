import torch
import random
import numpy as np
import pandas as pd
import os
import logging
import argparse
from models.transliteration_model import TransliterationModel
from dataloader.dataloader import TransliterationDataset
from utils.functions import init_optimizer, set_seed
import gc

logging.basicConfig(format='%(asctime)s %(message)s: ', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def main(args):

    set_seed(args)
        
    df = pd.read_csv(args.train_file)
    df.columns = ['source', 'target']

    train_src_sentences = df['source'].values.tolist()
    train_tgt_sentences = df['target'].values.tolist()

    df = pd.read_csv(args.val_file)
    df.columns = ['source', 'target']

    valid_src_sentences = df['source'].values.tolist()
    valid_tgt_sentences = df['target'].values.tolist()
        
        
    val_data_size = len(valid_src_sentences)
    data_offset = len(train_src_sentences) - val_data_size
    
    counter = 0
    bst_cer = 0
    bst_epoch = 0
    
    good_data = {}
    
    for i in range(0, len(train_src_sentences), args.train_step_size):
        
        logging.info(f"Training on Chunk #### {counter} ")
                
        train_data = train_src_sentences[i : i +  args.train_step_size], train_tgt_sentences[i : i +  args.train_step_size]
        valid_data = valid_src_sentences, valid_tgt_sentences
        
        # convert the list to pandas dataframes
        train_df = pd.DataFrame()
        train_df['src']  = train_data[0]
        train_df['tgt']  = train_data[1]
        
        valid_df = pd.DataFrame()
        valid_df['src']  = valid_data[0]
        valid_df['tgt']  = valid_data[1]
        
        load_weights = True if i else False
        
        save_folder = f"{args.lang}_transformer_model"
                
        transliteration_model = TransliterationModel(args,  train_df, valid_df, 
                                                     chunk = 0, checkpoint_folder = save_folder, 
                                                     load_weights = load_weights, best_metrics = (bst_epoch, bst_cer))
        
        # Train model
        transliteration_model.train_model(chunk = 0)
        _ = gc.collect()
        
        bst_epoch, bst_cer = transliteration_model.get_best_metrics()
        
        samples = transliteration_model.get_good_samples()
        good_data.update(samples)
                 
        counter += 1
        
        if counter == args.chunks:
            break
            
    # save good samples
    data = pd.DataFrame(list(good_data.items()), columns = ['source', 'target'])
    data.to_csv(f'../data/KB_{args.lang}.csv', index = False)
    print('Finished.')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train the transliteration transformer model,\
                                         if there is a model already trained, this command would replace it")

    parser.add_argument("--epochs", help="number of training epochs", default=1000, type=int)
    parser.add_argument("--batch_size", help="Batch size. Default: 32", default= 32, type=int)
    parser.add_argument("--d_model", help="Transformer model dimension. Default 512", default=512, type=int)
    parser.add_argument("--train_file",type = str, default = None, help ="train text data file")
    parser.add_argument("--val_file", type = str, default = None, help  = "validation text data file")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--train_step_size", type = int , default = 1000, help = "step size used to chunk train data")
    parser.add_argument("--chunks", type = int, default  = 1, help = "number of chunks to use for train/validation")
    parser.add_argument("--token_embedding_type", type=str, default = "hashing", choices =['hashing', 'normal'], help = "token embedding  technique")
    parser.add_argument("--position_embedding_type", type=str, default = "canine", choices =["canine", 'attn_paper'], help="position encoding technique")
    parser.add_argument("--dropout", type= float, default=0.0 , help = "dropout rate")
    parser.add_argument("--attn_heads",default = 4, type = int, help ="number of attention heads" )
    parser.add_argument("--use_local_transformer" ,action="store_true", help = "use local transformer encoder")
    parser.add_argument("--activation_fn", type = str, default ="relu", choices = ["gelu", "relu"], help ="activation function")
    parser.add_argument("--transformer_ff_size", type = int , default = 2048, help = "transformer feed forward dimension")
    parser.add_argument("--max_len", type = int, default = 32, help = "maximum length of input sequence")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--patience", type = int, default = 5, help = "patience parameter to modulate early stopping")
    parser.add_argument("--use_label_smoothing", action="store_true", default =False, help = "toggle between using a using label smoothing or not")
    parser.add_argument("--smoothing_param", type = float, default = 0.0, help = "label smoothing parameter")
    parser.add_argument("--lang", type=str, choices = ["es_it", "es_pt"], help ="language type")

    args = parser.parse_args()
    
    main(args)