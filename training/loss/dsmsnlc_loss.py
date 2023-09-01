from math import sqrt

import torch
import torch.nn as nn
from einops import einops

from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC
from torch.nn import functional as F


@LOSSFUNC.register_module(module_name="dsmsnlc_loss")
class DSMSNLCLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()

    def dsmsnlc_loss(self, nlc, mask):
        size = int(sqrt(nlc.shape[1]))
        mask = F.adaptive_max_pool2d(mask, (size, size))
        mask = 1 - torch.abs(
            einops.rearrange(mask, 'b w h -> b (w h) 1') - einops.rearrange(mask, 'b w h -> b 1 (w h)'))
        loss = F.binary_cross_entropy(nlc, mask.float())
        return loss

    def forward(self, pred_label, pred_nlc, label, mask):
        loss = F.cross_entropy(pred_label, label)
        for i in pred_nlc.values():
            loss += self.dsmsnlc_loss(i, mask)
        return loss
