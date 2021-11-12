import sys
import os
import argparse
import torch.nn as nn
from models.transliteration_model import TransliterationModel
from dataloader.dataloader import TransliterationDataset, collate_fn
from torch.utils.data import DataLoader
from utils.functions import init_optimizer, set_seed
import json
import re

tokens2int = json.load("../../data/vocab.json")

def tokenize_in(test_sentences):
    
    return [[ tokens2int[i] for i in sentence ] for sentence in test_sentences]

def preprocess_text(self, text):
        """Preprocess incoming text for model
           Normalize text
        """

        #         text = text.replace('ß',"b")
        #         text = text.replace('à',"a")
        #         text = text.replace('á',"a")
        #         text = text.replace('ç',"c")
        #         text = text.replace('è',"e")
        #         text = text.replace('é',"e")
        #         text = text.replace('$',"s")
        

        text = text.lower()
        text = re.sub(r'[^A-Za-z0-9 ,!?.]', '', text)
        text = text.replace('(', '')
        text = text.replace(')', '')

        # Remove '@name'
        text = re.sub(r'(@.*?)[\s]', ' ', text)

        # Replace '&amp;' with '&'
        text = re.sub(r'&amp;', '&', text)

        # Remove trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        #         text = re.sub(r'([h][h][h][h])\1+', r'\1', text)
        #         text = re.sub(r'([a-g-i-z])\1+', r'\1', text)  #Remove repeating characters
        text = re.sub(r' [0-9]+ ', " ", text)
        text = re.sub(r'^[0-9]+ ', "", text)

        return text
    
def split(self, text):
    
    """Split sentences based on punctuation and spaces
       Store punctuation and known words (we don't need to predict words that exist in the dataset)
       Returns:
        Tuple: Splits of words to be passed through the model, and the removed words and their indexes
    """

    splits = re.findall(r"[\w']+|[?!.,]", text)

    to_be_added = []
    idx_to_be_added = []

    forbidden = ["?", "!", ".", ","] + self.known_idx

    for i, split in enumerate(splits):

        if split in forbidden:
            if split in self.known_idx:
                to_be_added.append(self.known[split])
            else:
                to_be_added.append(split)
            idx_to_be_added.append(i)

    splits = [i for i in splits if not i in forbidden]
    
    return splits, to_be_added, idx_to_be_added


def transliterate_phrase(self, text):
    """Transliterate phrase into batches of word using greedy search
       Args:
        text (str): Sentence, or a group of sentences separated by a period.
       Returns:
        str: Splits of words to be passed through the model, and the removed words and their indexes
    """

    text = text.replace("0","")
    text = text.replace("6","")

    #Get splits
    text = preprocess_text(text)
    phrase, to_be_added, idx_to_be_added = split(text.lower())

    result = []

    #Sometimes all the words in a sentence exist in the known dict
    #So the returned phrase is empty, we check for that
    if len(phrase) > 0: 

        max_len_phrase = max([len(i) for i in phrase])

        #Pad and tokenize sentences
        #Idea? Pad with random text serving as auxilliary input
        input_sentence = []
        for word in phrase:
            input_sentence.append([self.in_token_to_int[i] for i in word] + [self.out_token_to_int["<pad>"]]*(max_len_phrase-len(word)))

        #Convert to Tensors
        input_sentence = torch.Tensor(input_sentence).long().T.to(self.device)
        preds = [[self.out_token_to_int["<sos>"]] * len(phrase)] 

        #A list of booleans to keep track of which sentences ended, and which sentences did not
        end_word = len(phrase) * [False]
        src_pad_mask = (input_sentence == self.pad_token).transpose(0, 1)

        with torch.no_grad():

            output = self.model(input_ids = input_sentence, attention_mask = src_pad_mask)['output_token']

            while not all(end_word): #Keep looping till all sentences hit <eos>
                output_sentence = torch.Tensor(preds).long().to(self.device)

                output = self.model(input_ids = output_sentence, attention_mask = None)['output_token']

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

    #Re-add removed punctuation and words
    for item, idx in zip(to_be_added, idx_to_be_added):
        if item == "?":
            item = "؟"
        elif item == ",":
            item = "،"
        result.insert(idx, item)

    result = " ".join(result)
    return result


def transliterate_list(self, texts, step_size=200, progress_bar=True):
    """Transliterate a list of phrases into batches of word using greedy search, then join them together
       Args:
        list: List of phrases in source sentences.
       Returns:
        list: List of phrases converted into target language
    """
    results = []
    if len(texts) < step_size:
        step_size = len(texts)

    if progress_bar:
        iterator = tqdm(range(0, len(texts), step_size))
    else:
        iterator = range(0, len(texts), step_size)

    for i in iterator: 

        out = self.transliterate_phrase(" lkrb3 ".join(texts[i:i+step_size]))
        splitted_sentences = [ex.strip() for ex in out.split(" " + self.transliterate_phrase("lkrb3") + " ")]

        if len(splitted_sentences) != len(texts[i:i+step_size]):
            logging.error("DANGER, a problem happened in de-tokenization, change the splitting word")
            break

        results.extend(splitted_sentences)

    return results

def run_inference(args):
    
    test_src_sentences, test_tgt_sentences = [],[]
    
    if args.input_file:
        with open(args.input_file , 'r') as f:
            for line in f:
                if len(line.strip().split(" ||| ")) == 3:
                    _, src,tgt = line.strip().split(" ||| ")
                test_src_sentences.append(src.lower())
                test_tgt_sentences.append(tgt.lower())

        f.close()
        
    else:
        RuntimeError("Sorry empty file is not allowed!")
        
    # Test datasets
    test_src_sentences = list(map(lambda x : preprocess_text(x), test_src_sentences))
    test_tgt_sentences = list(map(lambda x : preprocess_text(x), test_tgt_sentences))     

    















if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description = "Run transliteration inference.")
    args.add_argument("--input_file", type = str, default = None,  help = "test file you want to transliterate")
    args.add_argument("--out_file", type = str, default = "transliterated_file.txt", help = "output transliteratiion for the input test file")
    args.add_argument("--seed", type = int , default =42, help = "random seed value")
    
    args = args.parse_args()
    
    run_inference(args)



