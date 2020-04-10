# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/9 15:19, matt '


from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


def count_time(prev_time, cur_time):
    h, reminder = divmod((cur_time-prev_time).seconds, 3600)
    m, s = divmod(reminder, 60)
    time_str = "time %02d:%02d:%02d" %(h, m, s)
    return time_str


def accuracy(output, target, k=1):
    batch_size = target.size(0)
    _, ind = output.topk(k, 1, True, True)
    correct = ind.eq(target.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item()*(100.0/batch_size)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FocalLoss(nn.Module):

    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class Display_board:
    def __init__(self, port=8097, viz=None, env_name=None):
        if viz is None:
            self.viz = Visdom(port=port, env=env_name)
        else:
            self.viz = viz

    def add_Line_windows(self, name, X=0, Y=0):

        w = self.viz.line(X=np.array([X]), Y=np.array([Y]), opts=dict(title=name))
        return w

    def update_line(self, w, X, Y):
        self.viz.line(X=np.array([X]), Y=np.array([Y]), win=w, update="append")

    def show_image(self, image, title="test"):
        # plt.imshow(image)
        self.viz.image(image, opts=dict(title=title))

    def show_heatmap(self, predict, title="test"):
        self.viz.heatmap(X=predict, opts=dict(title=title))
