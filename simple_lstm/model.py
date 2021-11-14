import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from loguru import logger

class LSTM(nn.Module):
    """
    Baseline LSTM model
    :params hparams : hyperparameter config
    :params n_classes : number of classes
    """
    
    def __init__(self, hparams, vocab_size = 67, n_classes = 10):
        super(LSTM, self).__init__()
        
        self.hparams = hparams
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size + 1, self.hparams.rnn_input_size)
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        self.lstm = nn.LSTM(input_size = self.hparams.rnn_input_size,
                            hidden_size = self.hparams.rnn_hidden_size,
                            num_layers = self.hparams.nlayers,
                            dropout = self.hparams.dropout_rate,
                            bidirectional = self.hparams.birnn,
                            batch_first = True
                           )
        self.fc = nn.Linear(self.hparams.rnn_hidden_size, self.n_classes)
        
    def forward(self, x):
        
        embeded = self.embedding(x)
        embeded_drop = self.dropout(embeded)
                
        # Propagate input through LSTM
        lstm_out, (lstm_hidden, _) = self.lstm(embeded_drop)
        lstm_hidden = lstm_hidden.permute(1,0,2)
        
        out = self.fc(lstm_hidden[:,-1,:])
        
        return out
    