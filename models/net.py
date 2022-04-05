import importlib
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, config):
      super(Network, self).__init__()
      # Network libraries chosen in config
      pre_network  = importlib.import_module('models.prenet.' + config['network']['prenet']).Network
      base_network = importlib.import_module('models.base.' + config['network']['base']).Network
    
      self.num_class = config['num_class']
      self.max_stack = config['max_stack']
      # Networks which distills the information in the image and computes segmentation
      self.prenet    = pre_network(config)
      self.net       = base_network(config)
      
      # Convolutional layer for classification and activation function
      self.classifier = nn.Conv2d(config['f'], config['num_class'], 1)
      self.ReLU  = nn.LeakyReLU(inplace=True)
        
      # Normalization layer
      if   config['normalization'] == 'batch':
        self.bn = nn.BatchNorm2d(config['num_class'])
      elif config['normalization'] == 'layerHW':
        self.bn = nn.LayerNorm([config['oup_dim'][0], config['oup_dim'][1]])
      elif config['normalization'] == 'layerCHW':
        self.bn = nn.LayerNorm([config['num_class'], config['oup_dim'][0], config['oup_dim'][1]])
      else:
        raise Exception('Not a valid normalization')
        
    def forward(self, x):
      init   = self.prenet(x)
      hidden = self.net(init)
      out    = self.ReLU(self.bn(self.classifier(hidden))).unsqueeze(1)
      return out
