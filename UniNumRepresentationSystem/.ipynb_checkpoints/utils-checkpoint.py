import torch
import torch.nn as nn
import re

#code from https://nlp.seas.harvard.edu/2018/04/03/attention.html#label-smoothing
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

    
def save_model(model_u, model_t, epoch, opt, in_token2int
               , out_token2int, out_int2token, hparams, save_path):
    """
    Saving model, TODO adding history
    """

    print("SAVE MODEL.")
            
    args = {
        'in_token2int': in_token2int,
        'out_token2int': out_token2int,
        'out_int2token' : out_int2token,
        'hparams' : hparams,
        'epoch': epoch,
        'model_U' : model_u,
        'model_T' : model_t,
        'model_U_state_dict': model_u.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
#         'metrics': metrics
    }
    torch.save(args, save_path)
    
def load_model(load_path, device):
    """
    Loading model
    args:
        load_path: string
    """
    checkpoint = torch.load(load_path, map_location = device)
    
    #     metrics = checkpoint['metrics'] TO DO 
    
    if 'hparams' in checkpoint: 
        hparams = checkpoint['hparams']


    in_token2int = checkpoint['in_token2int']
    out_token2int = checkpoint['out_token2int']
    out_int2token = checkpoint['out_int2token']
    
    model_u = checkpoint['model_U'].cuda()
        
    return model_u, in_token2int, out_int2token, out_token2int
    
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


def calculate_accuracy(y_hat, y_true):
    """calculate_accuracy
    Args:
        y_hat : model predicted character
        y_true : ground_truth character
    """
    y_hat = F.softmax(y_hat ,dim = -1)

    y_hat = torch.argmax(y_hat,  dim = -1)

    return (y_hat == y_true).sum().item() / y_true.size(0)
    

def tokenize_in(phrase, in_token_to_int, pad=True):

    tokenized = [in_token_to_int[i] for i in phrase.lower()]

    if pad:
        padded = tokenized + ( in_max - len(tokenized)) * [out_token_to_int["<pad>"]] 
    else: padded = tokenized

    return padded


def tokenize_out(phrase, out_token_to_int, pad=True):
    
    tokenized = [out_token_to_int["<sos>"]] + [out_token_to_int[i] for i in phrase] + [out_token_to_int["<eos>"]]

    if pad:
        padded = tokenized + (out_max - len(tokenized)) * [out_token_to_int["<pad>"]]
    else: padded = tokenized

    return padded