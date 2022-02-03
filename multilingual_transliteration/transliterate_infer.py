import sys
import os
import argparse
import torch
import torch.nn as nn
from models.transliteration_model import TransliterationModel
from dataloader.dataloader import TransliterationDataset, collate_fn
from torch.utils.data import DataLoader
from utils.functions import init_optimizer, set_seed
import json
import re
from tqdm import tqdm
from dataloader.custom_sampler import KSampler
import time
import torch.nn.functional as F
import logging
from utils.utils import calculate_cer, load_model
import numpy as np
import pandas as pd    
import pdb

#Keep numbers block
def split(text):
    """Split sentences based on punctuation and spaces
       Store punctuation and known words (we don't need to predict words that exist in the dataset)
       Returns:
        Tuple: Splits of words to be passed through the model, and the removed words and their indexes
    """

    splits = re.findall(r"[\w']+|[?!.,]", text)
    
    
    to_be_added     = []
    idx_to_be_added = []

    forbidden = ["?", "!", ".", ","]

    for i, split in enumerate(splits):
        if split in forbidden:
            idx_to_be_added.append(i)
            to_be_added.append(split)

    splits = [i for i in splits if not i in forbidden]

    return splits, idx_to_be_added, to_be_added


def preprocess_text(text):
    """Preprocess incoming text for model
       Normalize text
    """

    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9 ,!?.]', '', text)

    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'([h][h][h][h])\1+', r'\1', text)
    text = re.sub(r'([a-g-i-z])\1+', r'\1', text)  #Remove repeating characters
    text = re.sub(r' [0-9]+ ', " ", text)
    text = re.sub(r'^[0-9]+ ', "", text)

    return text


def transliterate_phrase(args, pretrained_utils, device, split_src_text, tgt_text):
    """Transliterate phrase into batches of word using greedy search
       Args:
        text (str): Sentence, or a group of sentences separated by a period.
       Returns:
        str: Splits of words to be passed through the model, and the removed words and their indexes
    """
    
    # unpack list of pretrained items
    model, in_token2int, out_int2token, out_token2int, token_embedder, position_embedder, transformer_encoder, transformer_decoder ,fc = pretrained_utils
    
#     split_src_text, idx_to_be_added, to_be_added = split(src_text)
            
    pad_token = 0
    eos_token = 2
    sos_token = 1
    
    result = []

    #Sometimes all the words in a sentence exist in the known dict
    #So the returned phrase is empty, we check for that
        
    if len(split_src_text) > 0: 
        
#         max_len_in_phrase = max([len(i) for i in split_src_text])

        #Pad and tokenize sentences
        #Idea? Pad with random text serving as auxilliary input
        input_sentence = []
        
        for word in split_src_text:
            input_sentence.append([in_token2int[i] for i in word]) #+ [out_token2int["<pad>"]]*(max_len_in_phrase-len(word)))
        
        input_sentence = torch.Tensor(input_sentence).long().T.to(device)
        
        preds = [out_token2int["<sos>"]] * len(split_src_text)
        end_word = len(split_src_text) * [False]
        
        src_mask = (input_sentence == out_token2int["<pad>"]).transpose(0,1)
                
        with torch.no_grad():
            
            src = position_embedder(token_embedder(input_sentence))
            mem = transformer_encoder(src, None, src_mask)
            
            while not all(end_word):
                
                output_sentence = torch.Tensor(preds).long().to(device)
                
                tgt = position_embedder(token_embedder(output_sentence))
                
                tgt_mask = (tgt == out_token2int["<pad>"]).transpose(0,1)                
                output = transformer_decoder(tgt = tgt, memory = mem, memory_key_padding_mask = src_mask)
        
                output =fc(output)
        
                output = output.argmax(-1)[-1].cpu().detach().numpy()
                                
                preds.append(output.tolist())

                end_word = (output == out_token2int["<sos>"]) | end_word  #Update end word states

                    
        preds = np.array(preds).T  #(words, words_len)
        
        for word in preds:  #De-tokenize predicted words
            tmp = []
            
            for i in word[1:]:
                if out_int2token[i] == "<eos>":
                    break
                tmp.append(out_int2token[i])
    
            result.append("".join(tmp))

    # Re-add removed punctuation and words
#     for item, idx in zip(to_be_added, idx_to_be_added):
#         result.insert(idx, item)

    result = " ".join(result)
        
    return src_text, tgt_text, result


def transliterate_list(args, pretrained_utils, texts, tgt_texts, step_size=1, progress_bar=True, device=None):
    """Transliterate a list of phrases into batches of word using greedy search, then join them together
       Args:
        list: List of phrases in source sentences.
       Returns:
        dataframe: DataFrame consisting of phrases from source language converted into target language
    """
    results = pd.DataFrame()
            
    res_tgt = []
    res_src = []
    true_tgt = []
    
    if len(texts) < step_size:
        step_size = len(texts)

    if progress_bar:
        iterator = tqdm(range(0, len(texts), step_size))
    else:
        iterator = range(0, len(texts), step_size)

    for i in iterator: 
                
        src_text, tgt_text, out = transliterate_phrase(args, pretrained_utils, device, " nan ".join(texts[i:i+step_size]),  " nan ".join(tgt_texts[i:i+step_size]))
                                
        splitted_sentences = [ex.strip() for ex in out.split(" " + transliterate_phrase(args, pretrained_utils, device, "nan","")[-1] + " ")]
        
        res_tgt.append("".join(splitted_sentences))
        res_src.append(src_text)
        true_tgt.append(tgt_text)

    results['source'] = res_src
    results['target'] = res_tgt
    results['gt_target'] = true_tgt
    
    return results

def run_inference(args, pretrained_utils, device):
    
    if  not args.pretrained:
        RuntimeError("You must provide pretrained model path")
    
    # set seed
    set_seed(args)
    
    #     test_src_sentences, test_tgt_sentences = [],[]

    #     if args.input_file:
    #         with open(args.input_file , 'r') as f:
    #             for line in f:
    #                 if len(line.strip().split(" ||| ")) == 3:
    #                     _, src,tgt = line.strip().split(" ||| ")
    #                 test_src_sentences.append(src.lower())
    #                 test_tgt_sentences.append(tgt.lower())

    #         f.close()

    #     else:
    #         RuntimeError("Sorry, empty file is not allowed!")
    
    df = pd.read_csv(args.input_file)
    df.columns = ['source', 'target']

    test_src_sentences = df['source'].values.tolist()
    test_tgt_sentences = df['target'].values.tolist()

    # Test datasets preprocessing
    test_src_sentences = list(map(lambda x : preprocess_text(x), test_src_sentences[:100]))
    test_tgt_sentences = list(map(lambda x : preprocess_text(x), test_tgt_sentences[:100]))
    
    
    test_transliterations_df = transliterate_list(args, pretrained_utils, test_src_sentences, test_tgt_sentences,step_size=1, progress_bar=True, device=device)
    
    true_ld = list(map(lambda data: calculate_cer(*data), list(zip(test_src_sentences, test_src_sentences))))
    
    
    gt_df = pd.DataFrame()
    gt_df['source'] = test_src_sentences
    gt_df['target'] = test_tgt_sentences
    gt_df['lev_distance'] = true_ld
    
    dist = list(map(lambda data: calculate_cer(*data), list(zip(test_transliterations_df['source'], test_transliterations_df['target']))))

    test_transliterations_df['lev_distance'] = dist
    
    logging.info("Saving ground-truth KB ....")
    gt_df.to_csv(f"{args.output_dir}/{args.lang}_ground_truth_knowledge_base.csv", index=False)
    
    return test_transliterations_df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Run transliteration inference.")
    parser.add_argument("--input_file", type = str, default = None,  help = "test file you want to transliterate")
    parser.add_argument("--seed", type = int , default =42, help = "random seed value")
    parser.add_argument("--batch_size", type = int ,default = 32, help = "test batch size")
    parser.add_argument("--output_dir", type=str, default="../data", help="path/to/output data directory")
    parser.add_argument("--lang", type=str, choices = ["es_it", "es_pt"], help ="language type")
    parser.add_argument("--pretrained", type = str, default = None, help= "path/to/pretrained model")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, in_token2int, out_int2token, out_token2int, token_embedder, position_embedder, transformer_encoder, transformer_decoder ,fc = load_model(args.pretrained, device= device)
        
    pretrained_utils = (model, in_token2int, out_int2token, out_token2int, token_embedder, position_embedder, transformer_encoder, transformer_decoder ,fc)

    df = run_inference(args, pretrained_utils, device)
    
    logging.info("Saving transliteration results ....")
    df.to_csv(f"{args.output_dir}/{args.lang}_results.csv", index=False)
    
    logging.info("Finished.")



