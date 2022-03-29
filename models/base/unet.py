import torch
import torchvision
from torch import nn
# import torch.nn.functional as F

    
class Network(nn.Module):
    def __init__(self, config):
#         super().__init_()
        super(Network, self).__init__()
        self.encoder    = Encoder(config['enc_chs'])
        self.decoder    = Decoder(config['dec_chs'])
        self.head       = nn.Conv2d(config['dec_chs'][-1], config['num_class'], 1)
        self.out_sz     = config['oup_dim']

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        return out

class Decoder(nn.Module):
    def __init__(self, chs):#=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
    
    
class Encoder(nn.Module):
    def __init__(self, chs):#=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs
    
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
#         self.relu  = nn.ReLU()
        self.relu  = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
