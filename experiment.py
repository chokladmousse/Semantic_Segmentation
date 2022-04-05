import os
from os.path import exists, dirname
import tqdm
import gc
import copy
import time
import importlib
import argparse
from datetime import datetime
import shutil

import torch
import torch.optim as optim

# from validation import validation

torch.cuda.empty_cache()

def parse_command_line():
    """
    Parse command line and get the name of the experiment and type
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, help='experiments name')
    parser.add_argument('-n', '--max_iters', type=int, default=100, help='max number of iterations')
    parser.add_argument('-r', '--reset', type=bool, default=False, help='Reset experiment when starting, default false')
    args = parser.parse_args()
    return args

def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    opt = config['opt']

    resume = os.path.join('exp', opt.exp)
    resume_file = os.path.join(resume, 'checkpoint.pt')
    
    opt.continue_exp = False
    if exists(resume_file):
        opt.continue_exp = True
    
    if opt.continue_exp and not opt.reset:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file, map_location=config['device'])
            
            config['inference'] = checkpoint['inference']
            config['network'] = checkpoint['network']

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            config['inference']['best_loss'] = checkpoint['best_loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)
    else:
        config['inference']['best_loss'] = float('inf')
    
    if opt.reset:
        open(os.path.join(resume, 'log'), 'w').close()
        open(os.path.join(resume, 'acc'), 'w').close()
        open(os.path.join(resume, 'IoU'), 'w').close()
        config['inference']['best_loss'] = float('inf')

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """
    from pytorch/examples
    """
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')

def save(config, filename='checkpoint.pt'):
    resume = os.path.join('exp', config['opt'].exp)
    resume_file = os.path.join(resume, filename)

    save_checkpoint({
            'state_dict': config['inference']['net'].state_dict(),
            'optimizer' : config['train']['optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
            'best_loss': config['inference']['best_loss'],
            'network': config['network'],
            'inference': {         
                'M': config['inference']['M'],
                'f': config['inference']['f'],
                'n': config['inference']['n'],
                'increase_ratio': config['inference']['increase_ratio'],
                'normalization': config['inference']['normalization'],
                'num_class': config['inference']['num_class'],
                'scale': config['inference']['scale'],
                'inp_dim': config['inference']['inp_dim'],
                'oup_dim': config['inference']['oup_dim'],
                'max_stack': config['inference']['max_stack'],
            },
        }, False, filename=resume_file)
    print("=> saved " + filename)

def train(train_func, data_func, config, post_epoch=None):
    while True:
        print('epoch: ', config['train']['epoch'])
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break
                
        exp_path = os.path.join('exp', config['opt'].exp)
        logger_loss = open(os.path.join(exp_path, 'log'), 'a+')
        logger_acc  = open(os.path.join(exp_path, 'acc'), 'a+')
        logger_IoU  = open(os.path.join(exp_path, 'IoU'), 'a+')
        for phase in ['train', 'valid']:
            generator = data_func(phase)
            
            print('start', phase, config['opt'].exp)
            
            loss = train_func(config, phase, generator)
            
            if phase == 'valid' and loss < config['inference']['best_loss']:
                config['inference']['best_loss'] = loss
                save(config, filename='best_checkpoint.pt')

        if config['train']['epoch'] >= config['opt'].max_iters:
            return
                
        config['train']['epoch'] += 1
        save(config)

        
def validation(train_func, data_func, config):
    phase = 'test'
    out = train_func(config, phase, data_func(phase))
        
        
def init():
    """
    task.__config__ contains the variables that control the training and testing
    make_network builds a function which can do forward and backward propagation
    """
    opt = parse_command_line()
    config = importlib.import_module('hyperparameters').__config__
    network = importlib.import_module('models.network')
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    exp_path = os.path.join('exp', opt.exp)
    try: os.makedirs(exp_path)
    except FileExistsError: pass

    config['opt'] = opt
    config['data_provider'] = importlib.import_module(config['data_provider'])
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=> Using", config['device'])
    
    func = network.make_network(config)
    
    reload(config)
    return func, config


if __name__ == "__main__":
    func, config = init()
    print("=> Loading data")
    data_func = config['data_provider'].init(config)
    train(func, data_func, config)
#     validation(func, data_func, config)

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
