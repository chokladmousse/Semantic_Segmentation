import importlib
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        base_network = importlib.import_module('models.base.' + config['network']['base']).Network
    
        self.num_class = config['num_class']
        self.max_stack = config['max_stack']
        self.init_net = base_network(config)
        
        self.conv = nn.Conv2d(3, config['f'], 3, padding=1)
        self.classifier = nn.Conv2d(config['f'], config['num_class'], 1)
        self.ReLU  = nn.LeakyReLU(inplace=True)
        
        if   config['normalization'] == 'batch':
            self.bn1 = nn.BatchNorm2d(config['f'])
            self.bn2 = nn.BatchNorm2d(config['num_class'])
        elif config['normalization'] == 'layer':
            self.bn1 = nn.LayerNorm([config['f'], config['oup_dim'][0], config['oup_dim'][1]])
            self.bn2 = nn.LayerNorm([config['num_class'], config['oup_dim'][0], config['oup_dim'][1]])
        else:
            raise Exception('Not a valid normalization')
        
    def forward(self, x):
        init   = self.ReLU(self.bn1(self.conv(x)))
        hidden = self.init_net(init)
        out    = self.ReLU(self.bn2(self.classifier(hidden))).unsqueeze(1)
        return out