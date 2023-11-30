import torch
import torch.nn as nn

"""
FocalLoss can be used for multi label classification problems, where a sample may contain multiple labels.
 In this case, the binary cross entropy loss function is usually used for training, but when the training samples are imbalanced,
  the binary cross entropy loss function may lead to difficulties in training. 
  FocalLoss is a loss function for imbalanced data, 
  which can alleviate the problem of imbalanced samples and focus more on difficult to classify samples during training. 
  For multi label classification problems, FocalLoss can be achieved by applying a binary cross entropy loss function to each label and summing the loss values of all labels. 
"""


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt)**self.gamma * BCE_loss
        if self.alpha is not None:
            # balanced_weights = self.alpha[0] * targets + self.alpha[1] * (1 - targets)
            # targets = targets.long()
            balanced_weights = self.alpha[targets.long()] * (1 - targets) + (1 - self.alpha)[targets.long()] * targets
            F_loss = balanced_weights * F_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

