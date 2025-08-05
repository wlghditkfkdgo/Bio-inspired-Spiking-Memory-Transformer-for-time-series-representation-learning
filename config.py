import os

# os.environ['CUBLAS_WORKSPACE_CONFIG'] =':4096:8'

import torch
import pandas as pd
from scipy.io import arff

from matplotlib import pyplot as plt
import numpy as np

import time
import argparse
import random

import torch.backends


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.random.initial_seed()  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-gpu
    torch.backends.cudnn.deterministic = True # reduce operation speed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.cuda.set_rng_state(seed)
    # torch.set_rng_state(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
def set_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# def parse_arguments(args):
def parse_arguments():
    
    parser = argparse.ArgumentParser(description='the hyperparameters for training')
    
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--config', dest='config', default=None, help='If you have config file, input the file path')
    parser.add_argument('--saved_epoch', dest='saved_epoch', nargs='+', type=int, default=[1, ], help='The epoch number saved as check point')
    
    save_arg = parser.add_argument_group("save information")
    save_arg.add_argument('--tag', dest='tag', nargs='+')
    save_arg.add_argument('-s', '--save', dest='best_save', action='store_true', help='Save best model state')
    save_arg.add_argument('--print_epoch', nargs='?', type=int, default=5, help='Epoch print infomation period')
    save_arg.add_argument('--log_dir', nargs='?', default='result', help='The directory name to save log file \n\t default: %(default)s')
    save_arg.add_argument('--save_patience', dest='save_log_patience', nargs='?', type=int, default=2)
    
    save_arg.add_argument('--test', dest='test', action='store_true', help='If true, model is tested after training.')
    save_arg.add_argument('--tsne', dest='tsne', action='store_true', help='If true, the emb of model is visualized.')
    save_arg.add_argument('--only_path_test', dest='only_path_test', action='store_true', help='If true, the time path of model is tested.')
    
    train_arg = parser.add_argument_group("training parameters")
    train_arg.add_argument('-nd', '--num_device', dest='num_device', nargs='?', type=int, choices=[0, 1, 2, 3], default=0, help='Device number : [0, 1, 2, 3] (default: %(default)s)')
    train_arg.add_argument('-e', '--epoch', dest='epoch', nargs='?', type=int, default=50, help='# of total epoch')
    train_arg.add_argument('--warm_up_epoch', dest='warm_up_epoch', nargs='?', type=int, default=100, help='# of total epoch for pre-training')
    train_arg.add_argument('-bs', '--batch_size', dest='batch_size', nargs='?', type=int, default=128, help='Batch size')
    train_arg.add_argument('-lr', '--learning_rate', dest='lr', nargs='?', type=float, default=1e-4, help='Learning rate')
    train_arg.add_argument('--max_lr', dest='max_lr', nargs='?', type=float, default=0.001, help='maximum learning rate in cosine lr scheduler')
    train_arg.add_argument('--scheduler', dest='scheduler', nargs='?', type=str, choices=['step', 'lambda', 'exponential', 'cosine', 'reduce'], default='reduce', help='scheduler (default: %(default)s)')
    train_arg.add_argument('--weight_decay', dest='weight_decay', nargs='?', type=int, default=6e-2)
    train_arg.add_argument('--num_workers', dest='num_workers', nargs='?', type=int, default=8)

    model_arg = parser.add_argument_group("Transformer parameters")
    model_arg.add_argument('--alpha', dest='alpha', nargs='?', type=float, default=0.1, help='hyperparameter for gating')
    model_arg.add_argument('-emb', '--embedding_dim', dest='embed_dim', nargs='?', type=int, default=256, help='d_model in transformer')
    model_arg.add_argument('-nh', '--num_heads', dest='num_heads', type=int, default=8, help='num of heads in self-attention layer')
    model_arg.add_argument('--layers', dest='num_layers', nargs='?', type=int, default=1, help='number of encoder layers')
    model_arg.add_argument('--time_layers', dest='time_num_layers', nargs='?', type=int, default=2, help='number of layers in neocortex')
    # model_arg.add_argument('--num_patches', dest='num_patches', type=int, default=64, help='number of patches in Data Stream (default: %(default)s)')
    model_arg.add_argument('--patch_size', dest='data_patch_size', type=int, default=16, help='patch size in data stream')
    model_arg.add_argument('--mlp_emb', dest='mlp_emb', type=int, default=None)
    model_arg.add_argument('--patch_stride', dest='stride', type=int, default=9)
    model_arg.add_argument('--connect_f', dest='connect_f', nargs='?', choices=['ADD', 'AND', 'IAND'], default='IAND', help='the types of residual connection in SNN')
    model_arg.add_argument('--gating', dest='gating', default='original', help='gating type')
    model_arg.add_argument('--attn', dest='attn', nargs='?', default='MSSA', choices=['SSA'],)
    snn_arg = parser.add_argument_group("snn parameters")
    snn_arg.add_argument('-T', '--time_steps', dest='time_steps', nargs='?', type=int, default=4, help='Total time steps for SNN simulation (default: %(default)s)')
    snn_arg.add_argument('-b', '--bias', dest='bias', action=argparse.BooleanOptionalAction, help='spiking neuron bias (option: `--no-bias`)')
    snn_arg.add_argument('-spk', '--spk_encoding', dest='spk_encoding', action='store_true', help='spike encoding')
    snn_arg.add_argument('--tau', dest='tau', type=float, default=2.0)
    
    data_arg = parser.add_argument_group("data arguments")
    data_arg.add_argument('--num_classes', dest='num_classes', type=int, default=5)
    data_arg.add_argument('--data_window_size', dest='window_size', type=int, default=187)
    data_arg.add_argument('--num_channels', dest='num_channels', type=int, default=1)
    data_arg.add_argument('--train_data_root', nargs='?', type=str, default='../data/5class_mitbih_train_7_187.csv', help='The root of train data root saved \n\t default: %(default)s')
    # data_arg.add_argument('--train_data_root', nargs='?', type=str, default='./data/aami_4cls_mitbih_train_denoise_180.csv', help='The root of train data root saved \n\t default: %(default)s')
    data_arg.add_argument('--test_data_root', nargs='?', type=str, default='../data/5class_mitbih_test_3_187.csv', help='The root of test data root saved \n\t default: %(default)s')
    data_arg.add_argument('--sampling', dest='sampling', choices=['avg', 'min', 'cut', 'smote', 'None'], default='None')
    data_arg.add_argument('--ratio', dest='train_val_ratio', nargs='+', type=float, default=None, help='ratio for splitting training and validation(test) dataset (default:%(default)s)')
    data_arg.add_argument('--n_folds', dest='n_folds', type=int, default=2)
    data_arg.add_argument('--best_folds', dest='best_folds', type=int, default=0)
    data_arg.add_argument('--segment_len', dest='segment_len', type=int, default=0)
    config = parser.parse_args()
    
    return config


class Config():
    def __init__(self):
        self.date = time.strftime("%y%m%d", time.localtime(time.time()))
    
    def set_args(self, args:argparse.ArgumentParser):
        
        tag = ""
        if args.tag is not None:
            for t in args.tag:
                tag = tag + f"+{t}"
                
        tag_args = ["data_patch_size", "time_num_layers", "embed_dim", "num_heads", "alpha", "warm_up_epoch", "epoch", "lr", "gating", "attn"]
        
        for ktag in tag_args:
            vtag = getattr(args, ktag)
            tag = tag + f"+{ktag}+{vtag}"
            
        self.save_result_path = os.path.join(os.getcwd(), args.log_dir, self.date, self.date + tag)
        
        self.save_log_path = os.path.join(self.save_result_path, 'log')
        os.makedirs(self.save_log_path, exist_ok=True)
        
        self.device = torch.device(f'cuda:{args.num_device}' if torch.cuda.is_available() else 'cpu')
        
        for karg, varg in args._get_kwargs():
            setattr(self, f"{karg}", varg)

        if self.best_save: 
            self.save_model_state_path = os.path.join(self.save_result_path, "model_state")
            os.makedirs(self.save_model_state_path, exist_ok=True) 
            
        if self.config : delattr(self, "config")
        delattr(self, "num_device")
        delattr(self, "log_dir")
        delattr(self, "tag")
        
    def load_args(self, config_path:str, config:argparse.ArgumentParser):
        
        args_info_path = os.path.join(config_path, "model_state", "config.pt")
        args = torch.load(args_info_path, map_location='cpu')
        
        for kargs, vargs in args.items():
            if hasattr(config, kargs) and kargs in ['test', 'only_path_test']:
                setattr(self, f"{kargs}", getattr(config, kargs))
            else:
                setattr(self, f"{kargs}", vargs)
                
        self.saved_epoch = config.saved_epoch if self.saved_epoch.__len__() < 3 else self.saved_epoch
            
        self.save_result_path = config_path
        self.save_log_path = os.path.join(self.save_result_path, 'log')
        self.device = torch.device(f'cuda:{config.num_device}' if torch.cuda.is_available() else 'cpu')
        self.save_model_state_path = os.path.join(self.save_result_path, "model_state")
    
    def save_arg(self):
        
        """call only in training phase
        """
        
        args = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        torch.save(args, self.save_model_state_path + "/config.pt")

    def print_info(self):

        args = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        print(f"{' PARAMETERS INFO ':=^100s}")

        with open(self.save_log_path + 'args.txt', 'w', encoding='utf-8') as args_txt:
            for k, v in args.items():
                arg = f"{k:-<30s}{str(v):->70s}"
                print(arg)
                args_txt.write(str(arg) + '\n')
                
            print('\n')
    