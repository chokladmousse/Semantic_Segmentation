import torch
from torch import nn
import numpy as np

class Network(nn.Module):
    def __init__(self, config):
      super(Network, self).__init__()
      # Resnet
      self.net  = HourglassBlock(config, config['f'], config['f'], int(config['f'] * config['increase_ratio']), config['oup_dim'], config['M'])
      
      # Correct amount of hidden channels
      self.conv = nn.Conv2d(2 * config['f'], config['f'], 3, padding=1)
      self.ReLU = nn.LeakyReLU(inplace=True)
        
      if   config['normalization'] == 'batch':
        self.bn = nn.BatchNorm2d(config['f'])
      elif config['normalization'] == 'layerHW':
        self.bn = nn.LayerNorm([config['oup_dim'][0], config['oup_dim'][1]])
      elif config['normalization'] == 'layerCHW':
        self.bn = nn.LayerNorm([config['f'], config['oup_dim'][0], config['oup_dim'][1]])
      else:
        raise Exception('Not a valid normalization')
        
    def forward(self, x):
      hidden = self.net(x)
      out    = self.ReLU(self.bn(self.conv(hidden)))
      return out

    
class HourglassBlock(nn.Module):
    def __init__(self, config, f_prev, f_in, f_out, dim, m):
      super(HourglassBlock, self).__init__()
        
      self.down = Block(f_in, f_out, dim, config['normalization'], down=True)
      self.pool = nn.MaxPool2d(2,2)
        
      newdim = [int(x / 2) for x in dim]
        
      if m > 1:
        self.inner = HourglassBlock(config, f_in, f_out, int(f_out * config['increase_ratio']), newdim, m-1)
        self.high = Block(2 * f_out, f_in, newdim, config['normalization'], down=False)
      else:
        self.inner = None
        self.high = Block(f_out, f_in, newdim, config['normalization'], down=True)
    
      self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
      self.alpha = nn.Parameter(torch.tensor(0.))
      self.conv  = nn.Conv2d(f_in, 2 * f_in, 1)
      self.ReLU  = nn.LeakyReLU(inplace=True)

      if   config['normalization'] == 'batch':
        self.bn = nn.BatchNorm2d(config['f'])
      elif config['normalization'] == 'layerHW':
        self.bn = nn.LayerNorm([dim[0], dim[1]])
      elif config['normalization'] == 'layerCHW':
        self.bn = nn.LayerNorm([config['f'], dim[0], dim[1]])
      else:
        raise Exception('Not a valid normalization')
        
    def forward(self, x):
      down = self.down(x)
      pool = self.pool(down)
        
      if self.inner is not None:
        inner = self.inner.forward(pool)
      else:
        inner = pool
        
      high = self.high(inner)
      up = self.up(high)

      residual = self.ReLU(self.bn(self.conv(x)))
        
      return residual + self.alpha * torch.cat((x,up),1)

class Block(nn.Module):
    def __init__(self, f_in, f_out, dim, normalization, down=None):
      super(Block, self).__init__()
      self.down = down
        
      self.conv1 = nn.Conv2d(f_in, f_out, 3, padding=1)
      self.conv2 = nn.Conv2d(f_out, f_out, 3, padding=1)
      self.conv3 = nn.Conv2d(f_out, f_out, 1)
        
      if f_in == f_out:
        self.skip_conv = None
      else:
        self.skip_conv = nn.Conv2d(f_in, f_out, 1)
        
      self.alpha = nn.Parameter(torch.tensor(0.))
      self.ReLU  = nn.LeakyReLU(inplace=True)
        
      if   normalization == 'batch':
        self.bn1 = nn.BatchNorm2d(f_out)
        self.bn2 = nn.BatchNorm2d(f_out)
        self.bn3 = nn.BatchNorm2d(f_out)
      elif normalization == 'layerHW':
        self.bn1 = nn.LayerNorm([dim[0], dim[1]])
        self.bn2 = nn.LayerNorm([dim[0], dim[1]])
        self.bn3 = nn.LayerNorm([dim[0], dim[1]])
      elif normalization == 'layerCHW':
        self.bn1 = nn.LayerNorm([f_out, dim[0], dim[1]])
        self.bn2 = nn.LayerNorm([f_out, dim[0], dim[1]])
        self.bn3 = nn.LayerNorm([f_out, dim[0], dim[1]])
      else:
        raise Exception('Not a valid normalization')
        
    def forward(self, x0):
      x1 = self.ReLU(self.bn1(self.conv1(x0)))
      x2 = self.ReLU(self.bn2(self.conv2(x1)))
      x3 = self.ReLU(self.bn3(self.conv3(x2)))
        
      if self.skip_conv is not None:
        residual = self.skip_conv(x0)
      else:
        residual = x0
        
      return residual + self.alpha * x3
