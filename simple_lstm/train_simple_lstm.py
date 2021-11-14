import os
import sys
import numpy as np
import argparse
import torch
import tensorflow as tf
import random

from model import LSTM
from dataloader import UninumTrainDataset, UninumTestDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from dataloader import collate_fn_pad
from torch import nn
from madgrad import MADGRAD
import time
import logging
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_optimizer(args, model):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == "madgrad":
        return MADGRAD(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    else:
        logger.info("Not a valid optimizer!")

def calc_acc(probs, targets):

    pred_num = torch.argmax(probs, axis = -1)
    acc = (pred_num == targets).sum().item()
    return acc
            

def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    
    train_dataset = UninumTrainDataset(args)
    
    ntrain = len(train_dataset)
    train_split = int(ntrain * (1 - args.val_ratio))
    val_split = ntrain - train_split
    
    train_ds, valid_ds = torch.utils.data.random_split(train_dataset, [train_split, val_split])
    test_ds  = UninumTestDataset(args)

    logger.info(f"train split is {train_split}, validation split is {val_split}, test size is {len(test_ds)}")
    

    train_dataloader = DataLoader(train_ds,
                                 batch_size = args.batch_size,
                                 shuffle = True,
                                 collate_fn = collate_fn_pad,
                                 num_workers = 6 
                                 )
    valid_dataloader = DataLoader(valid_ds,
                                 batch_size = args.batch_size,
                                 shuffle =False,
                                  collate_fn = collate_fn_pad,
                                  num_workers = 6
                                 )
    test_dataloader = DataLoader(test_ds,
                             batch_size = args.batch_size,
                             shuffle =False,
                              collate_fn = collate_fn_pad,
                              num_workers = 6
                             )
    
    if args.model_name == "lstm":
        model = LSTM(args)
        
    model.to(device)
    
    if args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    
    logging.info(model)
    
    opt = init_optimizer(args, model)
    
    global_step = 0
    valid_loss_min = float("inf")
    bad_epoch = 0
    
    
    for epoch in range(args.epochs):
        
        logging.info(f"Epochs {epoch}/{args.epochs}")
        
        train_loss, val_loss, train_acc, val_acc = 0,0,0,0
        train_loss_arr, valid_loss_arr = [],[]
        
        start_time = time.time()

        sys.stdout.flush()
                
        start_iter = 0
        
        logging.info("TRAIN")
        
        # Train
        model.train()
        
        pbar = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
                
        for i, (data) in enumerate(pbar,start = start_iter):
                        
            inp_seq , tgt, length = data
                                    
            inp_seq , tgt = inp_seq.to(device), tgt.to(device)
                        
            opt.zero_grad()

            pred_Y = model(inp_seq)
            
            outputs = F.softmax(pred_Y, dim=-1)
            loss = criterion(outputs, tgt)
                        
            loss.backward()
            opt.step()
            
            train_loss_arr.append(loss.item())
            acc = calc_acc(pred_Y, tgt)

            global_step += 1
            
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

            train_loss += loss.item() / tgt.size(0)
            train_acc += acc / tgt.size(0)
            
            pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} TRAIN ACCURACY:{:.2f}%".format(
            (epoch+1), train_loss, train_acc*100))
            logging.info("(Epoch {}) TRAIN LOSS:{:.4f} ACCURACY:{:.2f}%".format(
            (epoch+1), train_loss/(len(train_dataloader)), train_acc*100 / len(train_dataloader)))
            
        logging.info("VALID")
        
        # Evaluate
        model.eval()
        
        val_pbar = tqdm(iter(valid_dataloader), leave=True, total = len(valid_dataloader))
        
        for i , (data) in enumerate(val_pbar, start = start_iter):
            
            inp_seq , tgt, length = data
            
            inp_seq = inp_seq.to(device)
            tgt = tgt.to(device)
            
            pred_Y = model(inp_seq)
            
            outputs = F.softmax(pred_Y, dim=-1)
            
            loss = criterion(outputs, tgt)
            
            valid_loss_arr.append(loss.item())
            
            acc = calc_acc(outputs, tgt)
            
            val_loss += loss.item() / tgt.size(0)
            val_acc += acc / tgt.size(0)
            
            val_pbar.set_description("(Epoch {}) VAL LOSS:{:.4f} VAL ACCURACY:{:.2f}%".format(
            (epoch+1), val_loss, val_acc*100))
            logging.info("(Epoch {}) VAL LOSS:{:.4f} ACCURACY:{:.2f}%".format(
            (epoch+1), val_loss/(len(valid_dataloader)), val_acc*100 / len(valid_dataloader)))
            
        # Checkpoint best model

        curr_train_loss = np.mean(train_loss_arr)
        curr_valid_loss = np.mean(valid_loss_arr)

        logger.info("train loss is {:.6f}".format(curr_train_loss) +  " validation loss is {:.6f}".format(curr_valid_loss))

        if curr_valid_loss < valid_loss_min:
            valid_loss_min = curr_valid_loss
            bad_epoch = 0
            torch.save(model.state_dict(), os.path.join( args.model_save_path , args.model_name +'.pt' ))

        else:
            bad_epoch += 1
            if bad_epoch >= args.patience:
                logger.info("training stops at epoch {}".format(bad_epoch))
                break

    # Test
    logging.info("TEST")
    
    model.eval()
    
    test_loss, test_acc = 0, 0
    start_iter = 0
    
    test_predictions = []
    nb_classes = 10
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    with torch.no_grad():
        
        test_pbar = tqdm(iter(test_dataloader), leave=True, total = len(test_dataloader))
        
        test_loss_arr = []
        
        for i , (data) in enumerate(test_pbar, start = start_iter):

            inp_seq , tgt, length = data
            inp_seq, tgt = inp_seq.to(device), tgt.to(device)

            pred_Y = model(inp_seq)
            
            outputs = F.softmax(pred_Y, dim=-1)
            
            _, test_pred = torch.max(outputs, dim=1)
            
            for t, p in zip(tgt.view(-1), test_pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            loss = criterion(outputs, tgt)
            
            test_loss_arr.append(loss.item())

            test_loss += loss.item() / tgt.size(0)
            test_acc += acc / tgt.size(0)
            acc = calc_acc(pred_Y, tgt)

            test_pbar.set_description("(Epoch {}) TEST LOSS:{:.4f} TEST ACCURACY:{:.2f}%".format(
            (epoch+1), test_loss, test_acc*100))
            logging.info("(Epoch {}) TEST LOSS:{:.4f} ACCURACY:{:.2f}%".format(
            (epoch+1), val_loss/(len(test_dataloader)), test_acc*100 / len(test_dataloader)))
        
        cur_test_loss = np.mean(test_loss_arr)
        logger.info("test loss is {:.6f}".format(cur_test_loss))
        logger.info(f"Confusion Matrix : {confusion_matrix}")
        
        #per-class accuracy
        logger.info(f"Per-class Accuracy : {confusion_matrix.diag() / confusion_matrix.sum(1)}")

        
def get_parser():
    parser = argparse.ArgumentParser(description="Universal Numeric Classification System")
    
    parser.add_argument("--train_dir", type= str, default = None, help ="train data directory")
    parser.add_argument("--test_dir",type=str, default=None, help="test data directory")
    parser.add_argument("--tokenizer_path", type=str, default= None, help ="tokenizer directory")
    parser.add_argument("--rnn_input_size", type = int, default = 3, help = "rnn input size")
    parser.add_argument("--rnn_hidden_size", type = int, default = 5, help="rnn hidden size")
    parser.add_argument("--nlayers", type = int, default = 2, help ="number of rnn layers")
    parser.add_argument("--birnn", default=True, help="BiLSTM")
    parser.add_argument("--char_level", default = True, help="character-level tokenization")
    parser.add_argument("--feats",type =str, default = "char", choices = ['char', 'embed'], help="feature type to use")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="percentage of train data for validation")
    
    parser.add_argument("--model_name", type=str, default="lstm", choices =["lstm", "conv_lstm", "char_former"], help="name of the model")
    parser.add_argument("--load_ckpt", default=False, help="load pretrained checkpoints")
    parser.add_argument("--epochs", default = 1000, type=int, help="epochs")
    parser.add_argument("--batch_size", default = 32, type = int, help = "sample batch size")
    parser.add_argument("--patience", default=5, type=int, help="patience regularizer")
    parser.add_argument("--dropout_rate", default =0.2, type=float, help="dropout rate")
    parser.add_argument("--seed", default = 42, type=int, help="random seed")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "madgrad"], help="optimizer type")
    parser.add_argument("--loss_type", type=str, default="cross_entropy", choices =["cross_entropy"], help="loss function definition")
    parser.add_argument("--lr", type =float, default = 0.001, help="set learning rate")
    parser.add_argument("--weight_decay", type= float, default = 1e-4, help = "weight decay parameter")
    parser.add_argument("--clip", default = False, help ="clip gradient")
    parser.add_argument("--max_norm", type =float, default=0.0, help="maximum value to clip gradient at")
    parser.add_argument("--smoothing_param", type=float, default =0.0, help="label smoothing parameter")
    
    parser.add_argument("--model_save_path", type=str, default=None, help= "best models output directory")
    
    return parser

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    main(args)