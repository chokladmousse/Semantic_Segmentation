import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.nn import CrossEntropyLoss

class DiceLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_class = config['num_class']
        self.inp_dim = config['inp_dim']
        self.oup_dim = config['oup_dim']
        self.scale = config['scale']
        
        self.cross_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, target, inputs):
        """
        input:  shape is (B, S, C, H, W)
        target: shape is (B, H, W)
        """
        
        # Cross-entropy loss
        cross_total = 0
        for i in range(inputs.shape[1]):
            cross_total += self.cross_loss(inputs[:,i], target)
        cross_total /= inputs.shape[1]
        
        # Dice loss
        no_zero = 1
        probs = F.softmax(inputs, 2)
        target = F.one_hot(target, num_classes=self.num_class).movedim(-1,1) # B, C, H, W
        target = target.unsqueeze(1) # B, 1, C, H, W
        
        overlap = probs*target              # B, S, C, H, W
        overlap = torch.sum(overlap,dim=-1) # B, S, C, H
        overlap = torch.sum(overlap,dim=-1) # B, S, C
        
        in_sum = probs*probs              # B, S, C, H, W
        in_sum = torch.sum(in_sum,dim=-1) # B, S, C, H
        in_sum = torch.sum(in_sum,dim=-1) # B, S, C
        
        #tr_sum = target*target#          # B, 1, C, H, W
        tr_sum = torch.sum(target,dim=-1) # B, 1, C, H
        tr_sum = torch.sum(tr_sum,dim=-1) # B, 1, C
        
        dice= 1 - (2*overlap + no_zero)/(tr_sum + in_sum + no_zero) # B, S, C
        dice_total = torch.mean(dice)
        
        return dice_total + cross_total