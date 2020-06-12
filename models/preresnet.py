### dropout has been removed in this code. original code had dropout#####
## https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules.patchup import PatchUp, PatchUpMode
from modules.drop_block import DropBlock
from utility.utils import to_one_hot
from modules.mixup import mixup_process, get_lambda
from modules.cutmix import CutMix
from data_loader import per_image_standardization
import random

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    """
    In this implementation PreActResNet consist of a Convolutional module followed
    by 4 residual blocks and a fully connected layer for classification.
    """
    def __init__(self, block, num_blocks, initial_channels, num_classes, per_img_std= False, stride=1, drop_block=7,
                 keep_prob=.9, gamma=.9, patchup_block=7):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        self.keep_prob = keep_prob
        self.gamma = gamma
        self.patchup_block = patchup_block
        self.dropblock = DropBlock(block_size=drop_block, keep_prob=keep_prob)
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.patchup_0 = PatchUp(block_size=self.patchup_block, gamma=self.gamma)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.patchup_1 = PatchUp(block_size=self.patchup_block, gamma=self.gamma)
        self.layer2 = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
        self.patchup_2 = PatchUp(block_size=5, gamma=self.gamma)
        self.layer3 = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
        self.patchup_3 = PatchUp(block_size=3, gamma=self.gamma)
        self.layer4 = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)
        self.patchup_4 = PatchUp(block_size=3, gamma=self.gamma)
        self.linear = nn.Linear(initial_channels*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def compute_h1(self,x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        return out

    def compute_h2(self,x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward(self, x, target= None, mixup=False, manifold_mixup=False, alpha=None,
                lam=None,
                patchup=False, dropblock=False, epoch=None, patchup_type=PatchUpMode.SOFT, k=2, dropblock_all=False):

        if self.per_img_std:
            x = per_image_standardization(x)

        lam_value = None

        if manifold_mixup or patchup:
            layer_mix = random.randint(0, k)
        elif dropblock and not dropblock_all:
            layer_mix = random.randint(1, k)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        out = x

        if alpha is not None and type(lam_value) == type(None):
            lam_value = get_lambda(alpha)
            lam_value = torch.from_numpy(np.array([lam_value]).astype('float32')).to(device)
            lam_value = Variable(lam_value)

        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)

        if layer_mix == 0 and patchup:
            cutmix = CutMix(beta=1.)
            target_a, target_b, out, portion = cutmix.apply(inputs=out, target=target)
            target_a = to_one_hot(target_a, self.num_classes)
            target_b = to_one_hot(target_b, self.num_classes)
            target_reweighted = portion * target_a + (1.0 - portion) * target_b

        elif layer_mix == 0 and not patchup:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam_value)

        out = self.conv1(out)

        if not patchup and not dropblock and (layer_mix == 1 and layer_mix <= k):
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam_value)
        elif patchup and (layer_mix == 1 and layer_mix <= k):
            target_a, target_b, target_reweighted, out, portion = self.patchup_0(out, target_reweighted, lam=lam_value,
                                                                                 patchup_type=patchup_type)

        if (dropblock and dropblock_all and 1 <= k) or (dropblock and layer_mix == 1 and layer_mix <= k):
            out = self.dropblock(out)

        out = self.layer1(out)

        if not patchup and not dropblock and layer_mix == 2 and layer_mix <= k:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam_value)
        elif patchup and layer_mix == 2 and layer_mix <= k:
            target_a, target_b, target_reweighted, out, portion = self.patchup_0(out, target_reweighted, lam=lam_value,
                                                                                 patchup_type=patchup_type)
        if (dropblock and dropblock_all and 2 <= k) or (dropblock and layer_mix == 2 and layer_mix <= k):
            out = self.dropblock(out)

        out = self.layer2(out)

        if not patchup and not dropblock and layer_mix == 3 and layer_mix <= k:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam_value)
        elif patchup and layer_mix == 3 and layer_mix <= k:
            target_a, target_b, target_reweighted, out, portion = self.patchup_0(out, target_reweighted, lam=lam_value,
                                                                                 patchup_type=patchup_type)

        if (dropblock and dropblock_all and 3 <= k) or (dropblock and layer_mix == 3 and layer_mix <= k):
            out = self.dropblock(out)

        out = self.layer3(out)

        if not patchup and not dropblock and layer_mix == 4 and layer_mix <= k:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam_value)
        elif patchup and layer_mix == 4 and layer_mix <= k:
            target_a, target_b, target_reweighted, out, portion = self.patchup_0(out, target_reweighted, lam=lam_value,
                                                                                 patchup_type=patchup_type)

        if (dropblock and dropblock_all and 4 <= k) or (dropblock and layer_mix == 4 and layer_mix <= k):
            out = self.dropblock(out)

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if target is not None:
            if patchup:
                return target_a, target_b, target_reweighted, out, portion
            return out, target_reweighted
        else:
            return out

    def get_layer_mix_lam(self, lam, lam_selection, max_rank_glb, k):
        lam_value = None

        if type(lam) == type(None):
            layer_mix = random.randint(1, k)
        else:
            if max_rank_glb:
                data, layer_mix = torch.max(lam[0][:k], 0)
            else:
                data, layer_mix = torch.min(lam[0][:k], 0)

            layer_mix = layer_mix.item() + 1

            if lam_selection:
                lam_value = data
                lam_value = torch.from_numpy(np.array([lam_value]).astype('float32')).to(device)
                lam_value = Variable(lam_value)
        return lam_value, layer_mix


def preactresnet18(num_classes=10, dropout = False, per_img_std = False, stride=1, drop_block=7, keep_prob=.9,
                   gamma=.9, patchup_block=7, patchup_prob=.7):
    return PreActResNet(PreActBlock, [2,2,2,2], 64, num_classes, per_img_std, stride= stride, drop_block=drop_block,
                        keep_prob=keep_prob, gamma=gamma, patchup_block=patchup_block)

def preactresnet34(num_classes=10, dropout = False, per_img_std = False, stride=1, drop_block=7, keep_prob=.9, gamma=.9,
                   patchup_block=7, patchup_prob=.7):
    return PreActResNet(PreActBlock, [3,4,6,3], 64, num_classes, per_img_std, stride= stride, drop_block=drop_block,
                        keep_prob=keep_prob, gamma=gamma, patchup_block=patchup_block)
