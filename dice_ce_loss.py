import torch.nn as nn
from dice_loss import DiceLoss

class DiceCELoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_ce=1.0):
        super(DiceCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.weight_dice * dice + self.weight_ce * ce
