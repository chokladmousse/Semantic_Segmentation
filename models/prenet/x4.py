import importlib
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, config):
      super(Network, self).__init__()
      middle_dim        = [int(x/2) for x in config['inp_dim']]
      config['oup_dim'] = [int(x/2) for x in middle_dim]
      config['scale']   = 4

      self.conv1 = nn.Conv2d(          3, config['f'], 5, padding=2)
      self.conv2 = nn.Conv2d(config['f'], config['f'], 5, padding=2)
      self.conv3 = nn.Conv2d(config['f'], config['f'], 5, padding=2)
      self.conv4 = nn.Conv2d(config['f'], config['f'], 5, padding=2)
      self.ReLU  = nn.LeakyReLU(inplace=True)
      self.pool1 = nn.MaxPool2d(2,2)
      self.pool2 = nn.MaxPool2d(2,2)
        
      if   config['normalization'] == 'batch':
        self.bn1 = nn.BatchNorm2d(config['f'])
        self.bn2 = nn.BatchNorm2d(config['f'])
        self.bn3 = nn.BatchNorm2d(config['f'])
        self.bn4 = nn.BatchNorm2d(config['f'])
      elif config['normalization'] == 'layerHW':
        self.bn1 = nn.LayerNorm([config['inp_dim'][0], config['inp_dim'][1]])
        self.bn2 = nn.LayerNorm([config['inp_dim'][0], config['inp_dim'][1]])
        self.bn3 = nn.LayerNorm([middle_dim[0], middle_dim[1]])
        self.bn4 = nn.LayerNorm([middle_dim[0], middle_dim[1]])
      elif config['normalization'] == 'layerCHW':
        self.bn1 = nn.LayerNorm([config['f'], config['inp_dim'][0], config['inp_dim'][1]])
        self.bn2 = nn.LayerNorm([config['f'], config['inp_dim'][0], config['inp_dim'][1]])
        self.bn3 = nn.LayerNorm([config['f'], middle_dim[0], middle_dim[1]])
        self.bn4 = nn.LayerNorm([config['f'], middle_dim[0], middle_dim[1]])
      else:
        raise Exception('Not a valid normalization')
        
    def forward(self, x0):
      x1 = self.ReLU(self.bn1(self.conv1(x0)))
      x2 = self.ReLU(self.bn2(self.conv2(x1)))
      y0 = self.pool1(x2)
      y1 = self.ReLU(self.bn3(self.conv3(y0)))
      y2 = self.ReLU(self.bn4(self.conv4(y1)))
      out = self.pool2(y2)
      return out
