import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
from loguru import logger


if __name__ == "__main__":
    
    data = pd.read_csv('../data/uninum.csv')
    data['target'] = data.index.tolist()
    
    data = pd.melt(data , id_vars = ['target'],value_name='text', var_name='lang', value_vars = data.columns.difference(['target']).tolist())
    
    tokenizer = Tokenizer(char_level = True)
    tokenizer.fit_on_texts(data.text.values)
        
    if not os.path.isdir("tokenizer"):
        os.mkdir('tokenizer')
        # save pretrained tokenizer
        torch.save(tokenizer, "../tokenizer/char_level_tok.pth")
        
        logger.info("Saved tokenizer.")
        
    else:
        tokenizer = torch.load("../tokenizer/char_level_tok.pth")
    
    train_df , test_df = train_test_split(data, test_size = 0.2, stratify = data.target)
    
    train_seqs = tokenizer.texts_to_sequences(train_df.text.values)
    test_seqs  = tokenizer.texts_to_sequences(test_df.text.values)
    
    train_seqs = pad_sequences(train_seqs, maxlen = None, padding='post', truncating='post')
    test_seqs =  pad_sequences(test_seqs,  maxlen = None, padding='post', truncating='post')
     
    logger.info(f"Train: {train_seqs.shape}" + f" Test: {test_seqs.shape}")
    logger.info(f"Vocab size : { len(tokenizer.word_index) }")
    
#     os.mkdir('tokenized_sequences')
    if os.path.isdir('tokenized_sequences'):
        assert len(os.listdir('tokenized_sequences')) != 0, 'empty directory'
        np.savez('../tokenized_sequences/train_seqs.npz', x = train_seqs, y = train_df.target.values )
        np.savez('../tokenized_sequences/test_seqs.npz',  x = test_seqs , y = test_df.target.values )
        logger.info("Saved tokenized sequences.")
    
    train_df.to_csv('../data/train_uninum_transformed.csv', index=False)
    test_df.to_csv('../data/test_uninum_transformed.csv',   index=False)