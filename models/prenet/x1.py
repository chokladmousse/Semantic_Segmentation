import importlib
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, config):
      super(Network, self).__init__()
      config['oup_dim'] = config['inp_dim']
      config['scale']   = 1

      self.conv = nn.Conv2d(3, config['f'], 3, padding=1)
      self.ReLU  = nn.LeakyReLU(inplace=True)
        
      if   config['normalization'] == 'batch':
        self.bn = nn.BatchNorm2d(config['f'])
      elif config['normalization'] == 'layerHW':
        self.bn = nn.LayerNorm([config['inp_dim'][0], config['inp_dim'][1]])
      elif config['normalization'] == 'layerCHW':
        self.bn = nn.LayerNorm([config['f'], config['inp_dim'][0], config['inp_dim'][1]])
      else:
        raise Exception('Not a valid normalization')
        
    def forward(self, x):
      out = self.ReLU(self.bn(self.conv(x)))
      return out
