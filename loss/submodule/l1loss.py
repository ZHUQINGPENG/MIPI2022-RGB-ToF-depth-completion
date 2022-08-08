"""
    Reference from Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    ======================================================================

    L1 loss implementation
"""


import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()

        self.args = args
        self.depth_valid = 0.0001

    def forward(self, pred, gt):

        mask = (gt > self.depth_valid).type_as(pred).detach()

        d = torch.abs(pred - gt) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
