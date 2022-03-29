import importlib
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        base_network = importlib.import_module('models.base.' + config['network']['base']).Network
    
        self.num_class = config['num_class']
        self.max_stack = config['max_stack']
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            base_network(config),
        ) for i in range(self.max_stack)] )
        
        self.conv = nn.Conv2d(3, config['f'], 3, padding=1)
        self.classifier = nn.Conv2d(config['f'], config['num_class'], 1)
        self.declassifier = nn.Conv2d(config['num_class'], config['f'], 1)
        self.alpha = nn.Parameter(torch.zeros(config['max_stack']-1))
        self.ReLU  = nn.LeakyReLU(inplace=True)
        
        if   config['normalization'] == 'batch':
            self.bn1 = nn.BatchNorm2d(config['f'])
            self.bn2 = nn.BatchNorm2d(config['num_class'])
            self.bn3 = nn.BatchNorm2d(config['f'])
        elif config['normalization'] == 'layer':
            self.bn1 = nn.LayerNorm([config['f'], config['oup_dim'][0], config['oup_dim'][1]])
            self.bn2 = nn.LayerNorm([config['num_class'], config['oup_dim'][0], config['oup_dim'][1]])
            self.bn3 = nn.LayerNorm([config['f'], config['oup_dim'][0], config['oup_dim'][1]])
        else:
            raise Exception('Not a valid normalization')

    def forward(self, x):
        out = torch.empty(self.max_stack, x.shape[0], self.num_class, x.shape[2], x.shape[3]).to(x.device)
        
        hidden = self.ReLU(self.bn1(self.conv(x)))
        
        for i in range(self.max_stack):
            if i == 0:
                hidden = self.hgs[i](hidden)
            else:
                hidden = self.hgs[i](residual + hidden)
            
            segmentation = self.ReLU(self.bn2(self.classifier(hidden)))
            if i < self.max_stack - 1:
                residual     = self.alpha[i] * self.ReLU(self.bn3(self.declassifier(segmentation)))
            out[i]       = segmentation
        
        out = out.swapaxes(0, 1)
        
        return out