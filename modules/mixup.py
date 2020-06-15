import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np


def mixup_process(out, target_reweighted, lam):
    """
    It can be use in Mixup and ManifoldMixup.
    https://arxiv.org/abs/1710.09412
    it applies mixup process for a mini-batch.
    Args:
        out: it is a samples in mini-batch or hidden representations
        for a mini-batch for Mixup or ManifoldMixup, respectively.
        target_reweighted: re-weighted target or interpolated targets in the mini-batch.
        iIn mixup it is the one hot embedding vector. And, in ManifoldMixup it is the re-weighted target
        calculated from previous layers.
        lam: the mixing interpolation policy coefficient.

    Returns:
        out: the interpolated of randomly selected pairs.
        target_reweighted: re-weighted target or interpolated targets in the mini-batch.
    """
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return out, target_reweighted


def get_lambda(alpha=1.0):
    """
    computes the interpolation policy coefficient in the mixup.
    Args:
        alpha: controls the shape of the Beta distribution.

    Returns:
        lam: a float number in [0, 1] that is the interpolation policy coefficient.
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam
