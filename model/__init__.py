# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/9 12:17, matt '

from model.resnet import resnet50, resnet101
from model.ArcFace import ArcMarginModel


def get_model(net_name):
    if net_name == "resnet101":
        return resnet101()
    elif net_name == "resnet50":
        return resnet50()
