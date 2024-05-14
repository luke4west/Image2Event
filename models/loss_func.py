import torch
import torch.nn as nn
import torch.nn.functional as F


def _neg_loss(pred, gt, alpha, beta):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""
    def __init__(self, alpha=2., beta=4.):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss
        self.alpha = alpha
        self.beta = beta

    def forward(self, out, target):
        return self.neg_loss(out, target, self.alpha, self.beta)