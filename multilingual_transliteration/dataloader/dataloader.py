import torch
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import numpy
import random

# Create a transliteration dataset class to 
# handle transliteration between any given pair of languages
class TransliterationDataset(Dataset):
    
    def __init__(self, src, target, src_texts, tgt_texts):
        
        self.src = src.values
        self.target = target.values
        self.src_texts= src_texts.values
        self.tgt_texts = tgt_texts.values
        self.src_lengths = [len(x) for x in self.src]
        self.tgt_lengths = [len(x) for x in self.target]
            
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):
        return (self.src[index], self.target[index], self.src_lengths[index], self.tgt_lengths[index], self.src_texts[index], self.tgt_texts[index])
      
    
def collate_fn(batch):
    
    src_word, tgt_word, src_len, tgt_len, src_text, tgt_text = zip(*batch)

    tensor_dim_1 = max(src_len)
    tensor_dim_2 = max(tgt_len)
    
    out_word = torch.full((len(src_word), tensor_dim_1), dtype=torch.long, fill_value=0)
    tgt_new  = torch.full((len(src_word), tensor_dim_2), dtype=torch.long, fill_value=0)

    for i in range(len(src_word)):
        
        out_word[i][:len(src_word[i])] = torch.LongTensor(src_word[i])
        tgt_new[i][:len(tgt_word[i])]  = torch.LongTensor(tgt_word[i])
    
    return (out_word, tgt_new, src_text, tgt_text)
    