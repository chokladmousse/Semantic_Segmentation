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
            return loss, combined_preds

def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']
    config['network'] = configs['network']
    
    network_lib = importlib.import_module('models.' + config['network']['model'])
    loss_func = importlib.import_module('loss_functions')
    
    network = network_lib.Network(config)
    forward_net = DataParallel(network.to(configs['device']))

    loss = loss_func.DiceLoss(config).to(configs['device'])
    
    def calc_loss(out, labels):
        return loss(out, labels)

    config['net'] = Trainer(forward_net, calc_loss)
    
    ## optimizer, experiment setup
    train_cfg['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad,config['net'].parameters()), train_cfg['learning_rate'])
#     train_cfg['optimizer'] = torch.optim.SGD(filter(lambda p: p.requires_grad,config['net'].parameters()), train_cfg['learning_rate'], momentum=train_cfg['momentum'], weight_decay=train_cfg['weight_decay'], nesterov=True)

    exp_path = os.path.join('exp', configs['opt'].exp)
    
    logger_loss = open(os.path.join(exp_path, 'log'), 'a+')
    logger_acc  = open(os.path.join(exp_path, 'acc'), 'a+')
    logger_IoU  = open(os.path.join(exp_path, 'IoU'), 'a+')
    
    scale = config['scale']
    n     = config['num_class']

    def make_train(config, phase, dataloader):
        optimizer = train_cfg['optimizer']
        net = config['inference']['net']
        net = net.train()
        
        total_loss = 0
        total_acc  = torch.zeros(config['inference']['max_stack'])
        total_IoU  = torch.zeros(config['inference']['max_stack'])
        if phase != 'test':
            for iter, (inputs, labels) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ascii=True):
                labels = labels.squeeze(1)
                # reset optimizer gradients
                optimizer.zero_grad()

                # both inputs and labels have to reside in the same device as the model's
                inputs = inputs.to(config['device'])
                labels = labels.to(device=config['device'], dtype=torch.int64)

                loss, combined_preds = net(inputs, labels)

                if phase == 'train':
                    # backpropagate
                    loss.backward()
                    # update the weights
                    optimizer.step()

                total_loss += loss.detach().item()
                
                acc, IoU = get_performance(combined_preds, labels[:,::scale,::scale], n)
                total_acc += acc
                total_IoU += IoU
            
            total_loss /= len(dataloader)
            total_acc  /= len(dataloader)
            total_IoU  /= len(dataloader)

            toprint = "\n" + phase + str(config['train']['epoch']) + " "
            logger_loss.write(toprint + str(total_loss))
            logger_acc.write(toprint)
            logger_IoU.write(toprint)
            logger_loss.flush()
            logger_IoU.flush()
            logger_acc.flush()

            for i in range(total_acc.size(0)):
              logger_acc.write(" " + str(acc[i].item()))
              logger_IoU.write(" " + str(IoU[i].item()))
              logger_acc.flush()
              logger_IoU.flush()

            return total_loss
        else:
            net = net.eval()
            
            for iter, (inputs, labels) in enumerate(dataloader):
                labels = labels.squeeze(1)
                
                result = net(inputs)

                acc, IoU = get_performance(result, labels[:,::scale,::scale])
                total_acc += acc
                total_IoU += IoU
            
            total_acc  /= len(dataloader)
            total_IoU  /= len(dataloader)

            print("Total accuracy on test set: " + str(total_acc))
            print("Mean IoU on test set:       " + str(total_IoU))

    return make_train

def get_performance(combined_preds, labels, n):
  labels = labels.unsqueeze(1)
  preds = combined_preds.argmax(2)

  acc = torch.mean(((preds == labels) & (labels != 9)).float(), dim=0).mean(dim=-1).mean(dim=-1).cpu()

  IoU = torch.empty(combined_preds.size(1), n-1)
  
  for i in range(n-1):
    pred_inds   = preds == i
    target_inds = labels == i

    intersection = (pred_inds & target_inds).sum(dim=0).sum(dim=-1).sum(dim=-1).cpu()
    union        = (pred_inds | target_inds).sum(dim=0).sum(dim=-1).sum(dim=-1).cpu()

    IoU[:,i] = intersection / union

  return acc, IoU.nanmean(dim=1)
