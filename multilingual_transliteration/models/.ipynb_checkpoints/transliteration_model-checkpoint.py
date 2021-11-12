import re
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
from utils.utils import calculate_cer
from utils.functions import init_optimizer, set_seed
from utils.utils import NoamOpt, AnnealingOpt, save_model, load_model,LabelSmoothing
from .position_embedding import PositionEmbedding, PositionalEncoding
from dataloader.dataloader import TransliterationDataset, collate_fn
from .local_self_attention import LocalSelfAttention
from .local_transformer_encoder_layer import LocalTransformerEncoderLayer
from .UniNumTransformerModel import UniversalNumericalTransformer, UniversalNumericalConfig
from dataloader.custom_sampler import KSampler
from sklearn.metrics import jaccard_score
import time
import pdb
import json
import gc
import json
from pprint import pprint


"""Create The Transformer Model using Local Attention Encoder
"""
class TransliterationModel():

    def __init__(self, args, train_df, valid_df,best_metrics, d_model=128, load_weights=False, checkpoint_folder=None, chunk : int = 1):

        """TransliterationModel.
        Create a TransliterationModel instance, then either train your own model
        or load pretrained weights. transliterate_list, and transliterate_phrase
        both functions that use the model for transliteration
        
        Args:
            d_model (int): Dimension of the transformer model.
            load_weights (bool): Wether to train the model or use a pre-trained one.
            checkpoint_folder (str): Directory of the model with "/".
            known (dict): Dictionary of the known transliterated words. 
            known_idx (list): List of the index of the known transliterated words.
        """
        
        self.args = args

        self.train_df = train_df
        self.valid_df = valid_df
        
        # Preprocess train words
        self.train_df['src'] = train_df.src.apply(lambda x : self.preprocess_text(x))
        self.train_df['tgt'] = train_df.tgt.apply(lambda x : self.preprocess_text(x))
        
        # Preprocess train words
        self.valid_df['src'] = valid_df.src.apply(lambda x : self.preprocess_text(x))
        self.valid_df['tgt'] = valid_df.tgt.apply(lambda x : self.preprocess_text(x))
        
        self.checkpoint_folder = checkpoint_folder
        self.d_model = d_model
        
        if os.path.isdir(checkpoint_folder):
            print(f"{checkpoint_folder} is an existing directory")
        else:
            print(f"creating new experiment {checkpoint_folder} folder ....")
            os.mkdir(checkpoint_folder)
            
        arg_path = os.path.join(checkpoint_folder, 'args.json')
        
        with open(arg_path, "w") as f:
            json.dump(vars(args), f)
        f.close()
    
        #Compute max sequence length for input and output
        self.in_max = self.train_df.apply(lambda x : len(x.src), axis =1).max()
        
        self.out_max =  self.train_df.apply(lambda x : len(x.tgt), axis =1).max() + 2  
        #Take into account eos and sos

        self.pad_token = 0
        self.eos_token = 2
        self.sos_token = 1
        
        self.best_epoch    = 0
        self.best_cer = 0
        self.good_teacher_samples = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"DEVICE: {self.device}")

        #Create the dictionary that maps each letter to it's corresponding embedding token
        #Input has one special token for padding
        in_tokens =  set(" ".join(self.train_df['src'].values.tolist())) | set(" ".join(self.valid_df['src'].values.tolist()))
        
        self.in_token_to_int = {token: (i+1) for i,token in enumerate(sorted(in_tokens))}

        self.in_token_to_int[0] = "<pad>"
                
        #Out put has three special tokens, <eos> <sos> and <pad>
        out_tokens = set(" ".join(self.train_df['tgt'].values.tolist())) | set(" ".join(self.train_df['tgt'].values.tolist()))
        
        self.out_token_to_int = {token: (i+3) for i,token in enumerate(sorted(out_tokens))}

        self.out_token_to_int["<pad>"] = self.pad_token
        self.out_token_to_int["<sos>"] = self.sos_token
        self.out_token_to_int["<eos>"] = self.eos_token

        self.out_int_to_token = {self.out_token_to_int[t]:t for t in self.out_token_to_int}
                
        # Instantiate encoder transformer model with default params
        self.model = UniversalNumericalTransformer(intoken = len(self.in_token_to_int), 
                                                   outtoken = len(self.out_token_to_int),
                                                   token_embedding_type = args.token_embedding_type,
                                                   position_embedding_type= args.position_embedding_type,
                                                   attention_heads = args.attn_heads,
                                                   use_local_transformer = args.use_local_transformer,
                                                   dropout = args.dropout,
                                                   activation =args.activation_fn,
                                                   transformer_ff_size = args.transformer_ff_size,
                                                   max_length = max(self.in_max, self.out_max)
                                                  ).to(self.device) 
        logging.info(f"Model {self.model}  created!")
        
        
        #Load model weights if pretrained
        if load_weights:
            self.model = torch.load(f"{checkpoint_folder}/chunk_{chunk}_cer_{best_metrics[1]}_epoch_{best_metrics[0]}_transliterate.pth", map_location=self.device)['model']
#             self.model = self.model.eval()
            logging.info(f"Model loaded from {checkpoint_folder}")
    
    def tokenize_row(self, row):
        """Tokenize a row from the dataset
        """

        x = row.copy()
        x.src = self.tokenize_in(x.src)
        x.tgt = self.tokenize_out(x.tgt)
        
        return x
    
    def tokenize_datasets(self):
        """Tokenize source language and target language words in df
        """

        logging.info(f"Tokenizing {self.train_df.shape[0]} word pairs in train data and {self.valid_df.shape[0]} word pairs in validation  data.")
        self.train_dataset = self.train_df.apply(lambda x: self.tokenize_row(x), axis=1)
        self.valid_dataset = self.valid_df.apply(lambda x: self.tokenize_row(x), axis=1)
        
    
    def tokenize_in(self, phrase : str = "nan", pad : bool =False):
        """Tokenize list of source sentences and pad it to maxlen
        """
        tokenized = [self.in_token_to_int[i] for i in phrase]
            
        if pad:
            padded = tokenized + (self.in_max - len(tokenized)) * [self.in_token_to_int["<pad>"]]
            
        else:
            padded = tokenized

        assert len(padded) != 0, "no tokenized sample"
        
        return padded


    def tokenize_out(self, phrase : str = "nan", pad : bool =False):
        """Tokenize list of target sentences and pad it to maxlen
        """
        tokenized = [self.out_token_to_int["<sos>"]] + [self.out_token_to_int[i] for i in phrase] + [self.out_token_to_int["<eos>"]]
        
        if pad:
            padded = tokenized + (self.out_max - len(tokenized))* [self.out_token_to_int["<pad>"]]
            
        else:
            padded = tokenized
            
        assert len(padded) != 0, "no tokenized samples"

        return padded

    
    def run_epoch(self,epoch, iterator, optimizer, criterion):
        """Perform one training epoch
        """
                
        self.model.train()
        
        pbar = tqdm(iter(iterator), leave=True, total=len(iterator))

        start_iter = 0
        total_loss = 0
        train_acc = []
        cer_distances = []
        
        good_samples = {}
        
        start_time = time.time()

        sys.stdout.flush()
                
        
        for i , (data) in enumerate(pbar, start = start_iter):
            
            src, trg, src_text ,tgt_text = data
            src = src.T.to(self.device)
            trg = trg.T.to(self.device)
            
            # make source pad mask
            src_mask = (src == self.pad_token).transpose(0,1)
            # make target pad mask
            tgt_mask = (trg[:-1, :] == self.pad_token).transpose(0,1)
            
            output = self.model(src, src_mask, trg[:-1, :], tgt_mask)['logits']
            
            output = output.reshape(-1, output.shape[2])                                    
            optimizer.optimizer.zero_grad()
            loss = criterion(output, trg[1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # transliterate source texts
            results = self.transliterate_list(src_text)
            
            # get correct samples
            for x_src, x_pred, x_tgt in list(zip(src_text, tgt_text, results)):
                if x_pred == x_tgt:
                    good_samples[x_src] = x_tgt
                    
                    # Generate attention labels for good samples
                    _, attention = self.model.compute_attention_soft_labels(x_src, x_tgt)
                    
                    y_attn = F.adaptive_avg_pool2d(attention, (1, 1))
        
            train_cer = list(map(lambda x : calculate_cer(*x), list(zip(results, tgt_text))))
            
            cer_distances.append(np.mean(train_cer))
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            
#             logging.info(f"Val batch --src {src_text} --tgt {tgt_text} --Model text {results}")
    
        return total_loss / len(iterator), np.mean(cer_distances), good_samples

    def run_validation(self,epoch, iterator, criterion):
        """Get validation loss
        """        
        self.model.eval()
        
        pbar = tqdm(iter(iterator), leave=True, total=len(iterator))
        
        val_acc = []
        cer_distances = []
        start_iter = 0
        total_loss = 0
        good_samples = {}
        
        start_time = time.time()
            
        sys.stdout.flush()
            
        for i, (data) in enumerate(pbar , start = start_iter):
            
            src, trg, src_text, tgt_text = data
            src = src.T.to(self.device)
            trg = trg.T.to(self.device)
            
            # make source pad mask
            src_mask = (src == self.pad_token).transpose(0,1)
            # make target pad mask
            tgt_mask = (trg[:-1, :] == self.pad_token).transpose(0,1)
        
            output = self.model(src, src_mask, trg[:-1, :], tgt_mask)['logits']
            
            output = output.reshape(-1, output.shape[2])

            loss = criterion(output, trg[1:].reshape(-1))
            total_loss += loss.item()
            
            # transliterate source texts
            results = self.transliterate_list(src_text)
            
            # get correct samples
            for x_src, x_pred, x_tgt in list(zip(src_text, tgt_text, results)):
                if x_pred == x_tgt:
                    good_samples[x_src] = x_tgt

                    
            val_cer = list(map(lambda x : calculate_cer(*x), list(zip(results, tgt_text))))
            cer_distances.append(np.mean(val_cer))
            
#             logging.info(f"Val batch --src {src_text} --tgt {tgt_text} --Model text {results}")
    
    
        return total_loss / len(iterator), np.mean(cer_distances), good_samples

    #Keep numbers block
    def split(self, text):
        """Split sentences based on punctuation and spaces
           Store punctuation and known words (we don't need to predict words that exist in the dataset)
           Returns:
            Tuple: Splits of words to be passed through the model, and the removed words and their indexes
        """

        splits = re.findall(r"[\w']+|[?!.,]", text)


        to_be_added = []
        idx_to_be_added = []

        forbidden = ["?", "!", ".", ","]

        for i, split in enumerate(splits):
            if split in forbidden:
                idx_to_be_added.append(i)
                to_be_added.append(split)

        splits = [i for i in splits if not i in forbidden]

        return splits, idx_to_be_added, to_be_added

    def transliterate_phrase(self, text):
        """Transliterate phrase into batches of word using greedy search
           Args:
            text (str): Sentence, or a group of sentences separated by a period.
           Returns:
            str: Splits of words to be passed through the model, and the removed words and their indexes
        """
    
        #Get splits
        phrase, to_be_added, idx_to_be_added = self.split(text.lower())
        
        # initiliaze transformer model functional modules
        pos_enc = self.model.get_position_encoder()
        tok_enc = self.model.get_token_embedder()
        encoder = self.model.get_transformer_encoder()
        decoder = self.model.get_transformer_decoder()
        fc      =  self.model.get_fc()
        
        result = []

        #Sometimes all the words in a sentence exist in the known dict
        #So the returned phrase is empty, we check for that
        if len(phrase) > 0: 

            max_len_phrase = max([len(i) for i in phrase])

            #Pad and tokenize sentences
            input_sentence = []
            for word in phrase:
                input_sentence.append([self.in_token_to_int[i] for i in word] + [self.out_token_to_int["<pad>"]]*(max_len_phrase-len(word)))

            #Convert to Tensors
            input_sentence = torch.Tensor(input_sentence).long().T.to(self.device)
            preds = [[self.out_token_to_int["<sos>"]] * len(phrase)] 

            #A list of booleans to keep track of which sentences ended, and which sentences did not
            end_word = len(phrase) * [False]
            src_pad_mask = (input_sentence == self.out_token_to_int["<pad>"]).transpose(0,1)

            with torch.no_grad():

                src = pos_enc(tok_enc(input_sentence))
                memory = encoder(src, None ,src_pad_mask)

                while not all(end_word): #Keep looping till all sentences hit <eos>
                    
                    output_sentence = torch.Tensor(preds).long().to(self.device)

                    trg = pos_enc(tok_enc(output_sentence))
                    output = decoder(tgt = trg, memory = memory, memory_key_padding_mask = src_pad_mask)
                    
                    output = fc(output)

                    output = output.argmax(-1)[-1].cpu().detach().numpy()
                    preds.append(output.tolist())

                    end_word = (output == self.out_token_to_int["<sos>"]) | end_word  #Update end word states
                    
                    if len(preds) > 50: #If word surpasses 50 characters, break out
                        break
                    
            preds = np.array(preds).T  #(words, words_len)

            for word in preds:  #De-tokenize predicted words
                tmp = []
                for i in word[1:]:   
                    if self.out_int_to_token[i] == "<eos>":
                        break
                    tmp.append(self.out_int_to_token[i])

                result.append("".join(tmp))
                
        result = " ".join(result)
        
        return result

    def transliterate_list(self, texts, step_size=1, progress_bar=True):
        """Transliterate a list of phrases into batches of word using greedy search, then join them together
           Args:
            list: List of phrases in Arabizi.
           Returns:
            list: List of phrases converted into Arabic script
        """
        results = []
        if len(texts) < step_size:
            step_size = len(texts)

        if progress_bar:
            iterator = tqdm(range(0, len(texts), step_size))
        else:
            iterator = range(0, len(texts), step_size)

        for i in iterator: 
            
            out = self.transliterate_phrase(" nan ".join(texts[i:i+step_size]))
            splitted_sentences = [ex.strip() for ex in out.split(" " + self.transliterate_phrase("nan") + " ")]
            
            results.extend(splitted_sentences)

        return results    
    
    
    def train_model(self, chunk : int = 0):
        """Train model for a specified number of epochs
        """

        logging.info("*"*30)
        logging.info("Training transliteration model ...")
        logging.info(f"{self.args.epochs} epochs, {self.args.batch_size} batch_size")
        logging.info("*"*30)
        
        self.tokenize_datasets()

        X_train = self.train_dataset['src']
        y_train = self.train_dataset['tgt']
        
        X_valid = self.valid_dataset['src']
        y_valid = self.valid_dataset['tgt']
        
                        
        train_set = TransliterationDataset(X_train, y_train, self.train_df['src'], self.train_df['tgt'])
        valid_set = TransliterationDataset(X_valid, y_valid, self.valid_df['src'], self.valid_df['tgt'])
                
        train_sampler = KSampler(train_set, self.args.batch_size)
        valid_sampler = KSampler(valid_set, self.args.batch_size)
        
        train_dataloader = DataLoader(train_set, sampler= train_sampler, 
                            batch_size= self.args.batch_size // self.args.gradient_accumulation_steps , collate_fn= collate_fn)
        valid_dataloader = DataLoader(valid_set, sampler = valid_sampler,
                                     batch_size = self.args.batch_size // self.args.gradient_accumulation_steps , collate_fn = collate_fn)
                
        logging.info("Create dataloaders.")
        
        if self.args.use_label_smoothing:
            criterion = LabelSmoothing(size= 10, padding_idx=self.out_token_to_int["<pad>"], smoothing= self.args.smoothing_param)
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=self.out_token_to_int["<pad>"])
        
        
        optimizer = NoamOpt(self.d_model, 1, 4000 ,optim.Adam(self.model.parameters(), lr=0))
        
        logging.info("Training...")
    
        min_loss = 99
        bad_epoch = 0
        best_epoch = 0
        best_cer_metric = float('inf')
        
        
        #Change model size
        for i in range(self.args.epochs):
            
            loss, train_cer, train_samples = self.run_epoch(i, train_dataloader, optimizer, criterion)
            
            if self.args.gradient_accumulation_steps > 1:
                loss /= self.args.gradient_accumulation_steps
            
            loss_val, val_cer, val_samples = self.run_validation(i, valid_dataloader, criterion)
            
            self.good_teacher_samples.update(val_samples)
            self.good_teacher_samples.update(train_samples)
            
            if loss_val < min_loss:
                min_loss = loss_val
                best_cer_metric = val_cer
                best_epoch = i
                
                # save best model
                save_model(self.model, i, optimizer, self.in_token_to_int,
                           self.out_token_to_int, self.out_int_to_token, self.args, 
                           self.checkpoint_folder + f"/chunk_{chunk}_cer_{best_cer_metric}_epoch_{best_epoch}_transliterate.pth")
                                
            else:
                bad_epoch += 1
                if bad_epoch >= self.args.patience:
                    break
            
            
            if i % 20 == 0 or i == (self.args.epochs - 1):
                print(f'EPOCH: {i} || Train Loss: {loss:.4f}, Train CER: {train_cer:.3f} || Val Loss: {loss_val:.4f},Val CER: {val_cer:.3f} || Good samples: {self.good_teacher_samples}')
                
        logging.info("="*20)
        logging.info("Training done, best loss %f, best cer %f" % (min_loss, best_cer_metric))
        logging.info("="*20)

        self.model = torch.load(self.checkpoint_folder + f"/chunk_{chunk}_cer_{best_cer_metric}_epoch_{best_epoch}_transliterate.pth")['model'].eval()
        self.best_cer = best_cer_metric
        self.best_epoch = best_epoch
        
    def get_best_metrics(self):
        return self.best_epoch, self.best_cer

    def get_good_samples(self):
        return self.good_teacher_samples

    def preprocess_text(self, text):
        
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
   