import importlib
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, config):
      super(Network, self).__init__()
      # Network libraries chosen in config
      pre_network  = importlib.import_module('models.prenet.' + config['network']['prenet']).Network
      base_network = importlib.import_module('models.base.' + config['network']['base']).Network
      
      # Network which distills the information in the image
      self.prenet = pre_network(config)

      # Parameters from the prenet and config
      self.dim       = config['oup_dim']
      self.num_class = config['num_class']
      self.max_stack = config['max_stack'] - 1

      # Networks which compute initial hidden state and iterates on the hidden state
      self.init_net = base_network(config)
      self.iter_net = base_network(config)
      
      # Convolutional layers needed for classification and making sure everything has the correct number of channels
      self.extractor  = nn.Conv2d(2 * config['f'], config['f'], 1)
      self.classifier = nn.Conv2d(config['f'], config['num_class'], 1)
      self.declassifier = nn.Conv2d(config['num_class'], config['f'], 1)

      # Residual connection alpha and activation function
      self.alpha = nn.Parameter(torch.tensor(0.))
      self.ReLU  = nn.LeakyReLU(inplace=True)
      
      # Normalization layers
      if   config['normalization'] == 'batch':
        self.bn1 = nn.BatchNorm2d(config['f'])
        self.bn2 = nn.BatchNorm2d(config['num_class'])
        self.bn3 = nn.BatchNorm2d(config['f'])
      elif config['normalization'] == 'layerHW':
        self.bn1 = nn.LayerNorm([config['oup_dim'][0], config['oup_dim'][1]])
        self.bn2 = nn.LayerNorm([config['oup_dim'][0], config['oup_dim'][1]])
        self.bn3 = nn.LayerNorm([config['oup_dim'][0], config['oup_dim'][1]])
      elif config['normalization'] == 'layerCHW':
        self.bn1 = nn.LayerNorm([config['f'], config['oup_dim'][0], config['oup_dim'][1]])
        self.bn2 = nn.LayerNorm([config['num_class'], config['oup_dim'][0], config['oup_dim'][1]])
        self.bn3 = nn.LayerNorm([config['f'], config['oup_dim'][0], config['oup_dim'][1]])
      else:
        raise Exception('Not a valid normalization')

    def forward(self, x):
      out = torch.empty(self.max_stack, x.shape[0], self.num_class, self.dim[0], self.dim[1]).to(x.device)
      
      # Distill input image
      hidden     = self.prenet(x)
      copy_input = hidden.clone()
      hidden_cat = torch.cat((copy_input, hidden), 1)
      hidden_con = self.ReLU(self.bn1(self.extractor(hidden_cat)))
      
      # Compute initial hidden state
      hidden = self.init_net(hidden_con)
      hidden_cat = torch.cat((copy_input, hidden), 1)
      hidden_con = self.ReLU(self.bn3(self.extractor(hidden_cat)))

      for i in range(self.max_stack):
        # Compute predicted segmentation from hidden state
        segmentation = self.ReLU(self.bn2(self.classifier(hidden)))
        residual     = self.ReLU(self.bn3(self.declassifier(segmentation)))
        out[i]       = segmentation

        # Add distilled input image to previous hidden state
        hidden_cat = torch.cat((copy_input, hidden), 1)
        hidden_con = self.ReLU(self.bn3(self.extractor(hidden_cat)))

        # Improve hidden state
        hidden = self.iter_net(residual + self.alpha * hidden_con)
        
      # Compute final predicted segmentation from hidden state
      segmentation = self.ReLU(self.bn1(self.classifier(hidden)))
      out[self.max_stack] = segmentation

      out = out.swapaxes(0, 1)
        
      return out
