# https://github.com/gentaiscool/end2end-asr-pytorch/blob/master/utils/optimizer.py
import torch
import torch.nn as nn
import Levenshtein as Lev
from models.UniNumTransformerModel import UniversalNumericalTransformer

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
    

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, min_lr=1e-5):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.min_lr = min_lr

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        "Implement `lrate` above"
        step = self._step
        return max(self.min_lr, self.factor * \
            (self.model_size ** (-0.5) * min(step **
                                             (-0.5), step * self.warmup ** (-1.5))))

class AnnealingOpt:
    "Optim wrapper for annealing opt"

    def __init__(self, lr, lr_anneal, optimizer):
        self.optimizer = optimizer
        self.lr = lr
        self.lr_anneal = lr_anneal
    
    def step(self):
        optim_state = self.optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / self.lr_anneal
        self.optimizer.load_state_dict(optim_state)
        
def calculate_cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence (hyp)
        s2 (string): space-separated sentence (gold)
    """
    return Lev.distance(s1, s2)

def save_model(model, epoch, opt, in_token2int
               , out_token2int, out_int2token, hparams, save_path):
    """
    Saving model, TODO adding history
    """

    print("SAVE MODEL.")
    
    transformer_encoder = model.get_transformer_encoder()
    transformer_decoder = model.get_transformer_decoder()
    position_embedder = model.get_position_encoder()
    token_embedder = model.get_token_embedder()
    fc = model.get_fc()
        
    args = {
        'in_token2int': in_token2int,
        'out_token2int': out_token2int,
        'out_int2token' : out_int2token,
        'hparams' : hparams,
        'epoch': epoch,
        'model' : model,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.optimizer.state_dict(),
        'transformer_encoder' : transformer_encoder,
        'transformer_decoder' : transformer_decoder,
        'token_embedder': token_embedder,
        'positon_embedder' : position_embedder,
        'fc' : fc,
        'optimizer_params': {
            '_step': opt._step,
            '_rate': opt._rate,
            'warmup': opt.warmup,
            'factor': opt.factor,
            'model_size': opt.model_size
        },
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
    
    token_embedder = checkpoint['token_embedder']
    position_embedder = checkpoint['positon_embedder']
    transformer_encoder = checkpoint['transformer_encoder'].cuda()
    transformer_decoder = checkpoint['transformer_decoder'].cuda()   

    model = checkpoint['model'].cuda()
    
    fc = checkpoint['fc']
    
    return model, in_token2int, out_int2token, out_token2int, token_embedder, position_embedder, transformer_encoder, transformer_decoder, fc