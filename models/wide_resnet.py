### dropout has been removed in this code. original code had dropout#####
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from utility.utils import to_one_hot
from modules.mixup import mixup_process, get_lambda
from modules.cutmix import CutMix
from data_loader import per_image_standardization
from modules.patchup import PatchUp, PatchUpMode
from modules.drop_block import DropBlock

act = torch.nn.ReLU()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(act(self.bn1(x)))
        out = self.conv2(act(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):

    def __init__(self, depth, widen_factor, num_classes, per_img_std=False, stride=1, drop_block=7, keep_prob=.9,
                 gamma=.9, patchup_block=7):
        super(Wide_ResNet, self).__init__()
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet_v2 depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.keep_prob = keep_prob
        self.gamma = gamma
        self.patchup_block = patchup_block
        self.dropblock = DropBlock(block_size=drop_block, keep_prob=keep_prob)
        self.conv1 = conv3x3(3, nStages[0], stride=stride)
        self.patchup_0 = PatchUp(block_size=self.patchup_block, gamma=self.gamma)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, stride=1)
        self.patchup_1 = PatchUp(block_size=self.patchup_block, gamma=self.gamma)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, stride=2)
        self.patchup_2 = PatchUp(block_size=5, gamma=self.gamma)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.patchup_3 = PatchUp(block_size=3, gamma=self.gamma)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)


    def forward(self, x, target=None, mixup=False, manifold_mixup=False, alpha=None,
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

        if layer_mix == 0 and patchup: # and patchup_type == MixingType.CUTMIX:
            cutmix = CutMix(beta=1.)
            target_a, target_b, out, portion = cutmix.apply(inputs=out, target=target)
            target_a = to_one_hot(target_a, self.num_classes)
            target_b = to_one_hot(target_b, self.num_classes)

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
            target_a, target_b, target_reweighted, out, portion = self.patchup_1(out, target_reweighted, lam=lam_value,
                                                                                 patchup_type=patchup_type)

        if (dropblock and dropblock_all and 2 <= k) or (dropblock and layer_mix == 2 and layer_mix <= k):
            out = self.dropblock(out)

        out = self.layer2(out)

        if not patchup and not dropblock and layer_mix == 3 and layer_mix <= k:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam_value)
        elif patchup and layer_mix == 3 and layer_mix <= k:
            target_a, target_b, target_reweighted, out, portion = self.patchup_2(out, target_reweighted, lam=lam_value,
                                                                                 patchup_type=patchup_type)

        if (dropblock and dropblock_all and 3 <= k) or (dropblock and layer_mix == 3 and layer_mix <= k):
            out = self.dropblock(out)

        out = self.layer3(out)

        if not patchup and not dropblock and layer_mix == 4 and layer_mix <= k:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam_value)
        elif patchup and layer_mix == 4 and layer_mix <= k:
            target_a, target_b, target_reweighted, out, portion = self.patchup_3(out, target_reweighted, lam=lam_value,
                                                                                 patchup_type=patchup_type)

        if (dropblock and dropblock_all and 4 <= k) or (dropblock and layer_mix == 4 and layer_mix <= k):
            out = self.dropblock(out)

        out = act(self.bn1(out))
        out = F.avg_pool2d(out, 8)
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


def wrn28_10(num_classes=10, dropout=False, per_img_std=False, stride=1, drop_block=7, keep_prob=.9, gamma=.9, patchup_block=7):
    # print ('this')
    model = Wide_ResNet(depth=28, widen_factor=10, num_classes=num_classes, per_img_std=per_img_std, stride=stride,
                        drop_block=drop_block, keep_prob=keep_prob, gamma=gamma, patchup_block=patchup_block)
    return model


def wrn28_2(num_classes=10, dropout=False, per_img_std=False, stride=1, patchup_block=7, drop_block=7, keep_prob=.9, gamma=.9):
    model = Wide_ResNet(depth=28, widen_factor=2, num_classes=num_classes, per_img_std=per_img_std, stride=stride,
                        drop_block=drop_block, keep_prob=keep_prob, gamma=gamma, patchup_block=patchup_block)
    return model

#PR