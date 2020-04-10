# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/9 13:58, matt '

import sys
sys.path.append("..")

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ArcMarginModel(nn.Module):
    def __init__(self, emb_size, easy_margin, margin_m, margin_s):
        super(ArcMarginModel, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(config.num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = margin_m
        self.s = margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(theta+m)
        phi = cosine*self.cos_m - sine*self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine>0, phi, cosine)
        else:
            phi = torch.where(cosine>self.th, phi, cosine-self.mm)

        one_hot = torch.zeros(cosine.size(), device=config.device)
        one_hot.scatter(1, label.view(-1, 1).long(), 1)
        output = (one_hot*phi) + ((1.0-one_hot)*cosine)
        output *= self.s
        return output


if __name__ == "__main__":
    net = ArcMarginModel(512, False, 0.5, 64)
    net.cuda()
    x = torch.randn((3, 512)).cuda()
    label = torch.tensor([1, 33, 222]).cuda()
    y = net(x, label)
    print(y.shape)