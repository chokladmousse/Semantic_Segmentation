import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, config):
      super(Network, self).__init__()
      self.net  = HourglassBlock(config, config['f'], int(config['f'] * config['increase_ratio']), config['oup_dim'], config['M'])

    def forward(self, x):
      return self.net(x)

class HourglassBlock(nn.Module):
    def __init__(self, config, f_in, f_out, dim, m):
      super(HourglassBlock, self).__init__()
      self.down = Block(f_in, f_out, dim, config['normalization'])
      self.pool = nn.MaxPool2d(2,2)
        
      newdim = [int(x / 2) for x in dim]
        
      if m > 1:
        self.inner = HourglassBlock(config, f_out, int(f_out * config['increase_ratio']), newdim, m-1)
      else:
        self.inner = None
        
      self.high = Block(f_out, f_in, newdim, config['normalization'])
      self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
      self.alpha = nn.Parameter(torch.tensor(0.))
        
    def forward(self, x):
      down = self.down(x)
      pool = self.pool(down)
      
      if self.inner is not None:
        inner = self.inner(pool)
      else:
        inner = pool
        
      high = self.high(inner)
      up = self.up(high)
        
      return x + self.alpha * up

class Block(nn.Module):
    def __init__(self, f_in, f_out, dim, normalization):
      super(Block, self).__init__()
        
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
