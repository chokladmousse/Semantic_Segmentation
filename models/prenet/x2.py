import importlib
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, config):
      super(Network, self).__init__()
      config['oup_dim'] = [int(x/2) for x in config['inp_dim']]
      config['scale']   = 2

      self.conv1 = nn.Conv2d(          3, config['f'], 5, padding=2)
      self.conv2 = nn.Conv2d(config['f'], config['f'], 5, padding=2)
      self.ReLU  = nn.LeakyReLU(inplace=True)
      self.pool  = nn.MaxPool2d(2,2)
        
      if   config['normalization'] == 'batch':
          self.bn1 = nn.BatchNorm2d(config['f'])
          self.bn2 = nn.BatchNorm2d(config['f'])
      elif config['normalization'] == 'layerHW':
          self.bn1 = nn.LayerNorm([config['inp_dim'][0], config['inp_dim'][1]])
          self.bn2 = nn.LayerNorm([config['inp_dim'][0], config['inp_dim'][1]])
      elif config['normalization'] == 'layerCHW':
          self.bn1 = nn.LayerNorm([config['f'], config['inp_dim'][0], config['inp_dim'][1]])
          self.bn2 = nn.LayerNorm([config['f'], config['inp_dim'][0], config['inp_dim'][1]])
      else:
        raise Exception('Not a valid normalization')
        
    def forward(self, x):
      y   = self.ReLU(self.bn1(self.conv1(x)))
      z   = self.ReLU(self.bn2(self.conv2(y)))
      out = self.pool(z)
      return out
