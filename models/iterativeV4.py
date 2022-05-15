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
      self.max_stack = config['max_stack']

      # Network which iterates on the hidden state
      self.iter_net = base_network(config)
      
      # Convolutional layers needed for classification and making sure everything has the correct number of channels
      self.extractor  = nn.Conv2d(2 * config['f'], config['f'], 1)
      self.classifier = nn.Conv2d(config['f'], config['num_class'], 1)
      self.declassifier = nn.Conv2d(config['num_class'], config['f'], 1)

      # Residual connection alpha and activation function
      self.alpha = nn.Parameter(torch.tensor(0.))
      self.beta  = nn.Parameter(torch.tensor(0.))
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
      initial_hidden = torch.zeros_like(hidden).to(x.device)
      
      # Concatenate distilled input image with initial hidden state and convolve to correct number of channels
      copy_input = hidden.clone()
      hidden_cat = torch.cat((copy_input, initial_hidden), 1)
      hidden_con = self.ReLU(self.bn1(self.extractor(hidden_cat)))
        
      for i in range(self.max_stack):
        # Improve hidden state
        temp_prev_hidden = hidden_con.clone()
        if i > 0:
          hidden = self.iter_net(self.beta * residual + self.alpha * hidden_con + prev_hidden)
        else:
          hidden = self.iter_net(self.alpha * hidden_con)
        
        # Compute predicted segmentation from hidden state
        segmentation = self.ReLU(self.bn2(self.classifier(hidden)))
        residual     = self.ReLU(self.bn3(self.declassifier(segmentation)))
        out[i]       = segmentation

        # Concatenate distilled input image with hidden state if at least one more iteration will happen
        if i < self.max_stack - 1:
          hidden_cat = torch.cat((copy_input, hidden), 1)
          hidden_con = self.ReLU(self.bn1(self.extractor(hidden_cat)))
          prev_hidden = temp_prev_hidden
      
      out = out.swapaxes(0, 1)
        
      return out
