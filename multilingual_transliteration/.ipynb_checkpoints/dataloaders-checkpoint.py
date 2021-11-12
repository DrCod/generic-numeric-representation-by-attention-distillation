import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import numpy
import random


# Create a transliteration dataset class to 
# handle transliteration between any given pair of languages
class TransliterationDataset(Dataset):
    
    def __init__(self, args):
        self.args = args
        self.get_source_sents(args)
        self.get_target_sents(args)
        self.src2tgt_tags = {}
        
        
    def get_source_sents(self, args):
        """get_source_sents returns a list of source sentences in the input file
         Args
             args : all parsed arguments
        """
        self.src_sentences = []
        with open(args.dataset, 'r') as f:
            for line in f:
                tag, src_txt, _ = line.split(" ||| ")
                
                self.src2tgt_tags.update(tag)

                self.src_sentences.append(src_txt)
        f.close()
        
        self.src_lengths = [len(sent) for sent in self.src_sentences]
        
        return
            
            
    def get_target_sents(self, args):
        """get_target_sents returns a list of target sentences in the input file
         Args
             args : all parsed arguments
        """
        self.target_sentences = []
        with open(args.dataset, 'r') as f:
            for line in f:
                tag,_,tgt_txt = line.split(" ||| ")
                
                self.src2tgt_tags.update(tag)
                self.src_sentences.append(tgt_txt)
        f.close()
        
        self.tgt_lengths = [len(sent) for sent in self.target_sentences]
        return
            
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, index):
        return (self.src_sentences[index], self.target_sentences[index], self.src_lengths[index], self.tgt_lengths[index])
    
    
    
def collate_fn(batch):
    src_word, tgt_word, src_len, tgt_len = zip(*batch)
    
    tensor_dim_1 = max(src_len)
    tensor_dim_2 = max(tgt_len)
    
    out_word = torch.full((len(src_word), tensor_dim_1), dtype=torch.long, fill_value=0)
    tgt_word = torch.full((len(src_word), tensor_dim_2), dtype=torch.long, fill_value=0)

    for i in range(len(src_word)):
        
        out_word[i][:len(src_word[i])] = torch.Tensor(src_word[i])
        tgt_word[i][:len(tgt_word[i])] = torch.Tensor(tgt_word[i])
    
    return (out_word, tgt_word)