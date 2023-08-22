from typing import Any

import torch.nn as nn

class AbstractLossClass(nn.Module):
    """Abstract class for loss functions."""
    def __init__(self):
        super(AbstractLossClass, self).__init__()

    def forward(self,  *input: Any):
        """
        Args:
            Loss input
            
        Return:
            loss: loss value
        """
        raise NotImplementedError('Each subclass should implement the forward method.')
