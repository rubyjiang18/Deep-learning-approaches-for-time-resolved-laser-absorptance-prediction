'''
Modified from 
https://github.com/nasa/pretrained-microscopy-models/blob/main/pretrained_microscopy_models/losses.py
'''

import torch.nn as nn
import torch.nn.functional as F

class DiceBCEWithActivationLoss(nn.Module):
    def __init__(self, weight=0.7, size_average=True):
        super(DiceBCEWithActivationLoss, self).__init__()
        self.weight = weight
        self.__name__ = 'DiceBCEWithActivationLoss'

    def forward(self, inputs, targets, smooth=1):
        # inputs are logits in range [-inf, +inf]
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        Dice_BCE = self.weight * BCE + (1-self.weight) * dice_loss
        
        return Dice_BCE