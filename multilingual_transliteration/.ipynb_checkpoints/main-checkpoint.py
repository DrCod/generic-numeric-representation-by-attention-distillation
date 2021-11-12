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

logging.basicConfig(format='%(asctime)s %(message)s: ', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def main(args):

    set_seed(args)
    
    train_src_sentences,valid_src_sentences, valid_tgt_sentences, train_tgt_sentences = [],[],[],[]
    
    with open(args.train_file, 'r') as f:
        for line in f:
            if len(line.strip().split(" ||| ")) == 3:
                _, src,tgt = line.strip().split(" ||| ") 
            train_src_sentences.append(src.lower())
            train_tgt_sentences.append(tgt.lower())

    f.close()
            
    with open(args.val_file, 'r') as f:
        for line in f:
            if len(line.strip().split(" ||| ")) == 3:
                _, src,tgt = line.strip().split(" ||| ") 
                
            valid_src_sentences.append(src.lower())
            valid_tgt_sentences.append(tgt.lower())

    f.close()
    

    train_data = train_src_sentences, train_tgt_sentences
    valid_data = valid_src_sentences, valid_tgt_sentences

    transliteration_model = TransliterationModel(args,  train_data, valid_data)

    # Train model
    transliteration_model.train_model()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train the transliteration transformer model,\
                                         if there is a model already trained, this command would replace it")

    parser.add_argument("--epochs", help="number of training epochs", default=100, type=int)
    parser.add_argument("--batch_size", help="Batch size. Default: 32", default= 32, type=int)
    parser.add_argument("--d_model", help="Transformer model dimension. Default 128", default=2048, type=int)
    parser.add_argument("--save_folder", help="Folder to save model in. Default: `checkpoint`", default="checkpoint")
    parser.add_argument("--train_file",type = str, default = None, help ="train text data file")
    parser.add_argument("--val_file", type = str, default = None, help  = "validation text data file")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")


    args = parser.parse_args()
    
    main(args)