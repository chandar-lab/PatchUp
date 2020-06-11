import numpy as np
import torch

from utility.utils import device


class CutMix(object):
    """
    The following code was implemented by the authors of CutMix paper.
    https://arxiv.org/abs/1905.04899
    following is the the link to the official implementation of CutMix.
    https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    """
    def __init__(self, beta):
        self.beta = beta

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def apply(self, inputs, target):
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(inputs.size()[0]).to(device)
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        # compute output
        return target_a, target_b, inputs, lam

#PR