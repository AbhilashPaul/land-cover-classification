import torch
import torch.nn as nn

# Define a Dice Loss implementation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).contiguous()
        
        # Apply softmax to inputs
        inputs = torch.nn.functional.softmax(inputs, dim=1)
        
        # Flatten the tensors
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Calculate intersection and union
        intersection = (inputs * targets_one_hot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss
