import os
import importlib
import tqdm

import torch
# import numpy as np
from torch import nn
from torch.nn import DataParallel
#from utils.misc import make_input, make_output, importNet

# import shutil

class Trainer(nn.Module):
    """
    The wrapper module that will behave differently for training or testing
    inference_keys specify the inputs for inference
    """
    def __init__(self, model, calc_loss=None):
        super(Trainer, self).__init__()
        self.model = model
        self.calc_loss = calc_loss

    def forward(self, inputs, labels):
        if not self.training:
            return self.model(inputs)
        else:
            combined_preds = self.model(inputs)
            loss = self.calc_loss(labels, combined_preds)
            return loss

def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']
    config['network'] = configs['network']
    
    network_lib = importlib.import_module('models.' + config['network']['model'])
    loss_func = importlib.import_module('loss_functions')
    loss = loss_func.DiceLoss(config)
    
    def calc_loss(out, labels):
        return loss(out, labels)
    
    network = network_lib.Network(config)
    forward_net = DataParallel(network.to(configs['device']))
    config['net'] = Trainer(forward_net, calc_loss)
    
    ## optimizer, experiment setup
    train_cfg['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad,config['net'].parameters()), train_cfg['learning_rate'])
#     train_cfg['optimizer'] = torch.optim.SGD(filter(lambda p: p.requires_grad,config['net'].parameters()), train_cfg['learning_rate'], momentum=train_cfg['momentum'], weight_decay=train_cfg['weight_decay'], nesterov=True)

    exp_path = os.path.join('exp', configs['opt'].exp)
    
    logger = open(os.path.join(exp_path, 'log'), 'a+')
    
    def make_train(config, phase, dataloader):
        optimizer = train_cfg['optimizer']
        net = config['inference']['net']
        net = net.train()
        
        total_loss = 0
        if phase != 'test':
            for iter, (inputs, labels) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ascii=True):
                labels = labels.squeeze(1)
                # reset optimizer gradients
                optimizer.zero_grad()

                # both inputs and labels have to reside in the same device as the model's
                inputs = inputs.to(config['device'])
                labels = labels.to(device=config['device'], dtype=torch.int64)

                loss = net(inputs, labels)
                
                if phase == 'train':
                    # backpropagate
                    loss.backward()
                    # update the weights
                    optimizer.step()

                toprint = "\n " + phase + "{}, iter{}, loss: {}".format(config['train']['epoch'], iter, loss.item())
                logger.write(toprint)
                logger.flush()
                
                total_loss += loss.detach().item()
            return total_loss / len(dataloader)
        else:
            net = net.eval()
            
            for iter, (inputs, labels) in enumerate(dataloader):
                labels = labels.squeeze(1)
                
                result = net(inputs)
                print(result.size())
                error(':)')
            
            # TODO: IoU and accuracy of result
            
            return out

    return make_train