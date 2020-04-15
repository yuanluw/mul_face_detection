# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/9 12:17, matt '

from model.resnet import *
from model.ArcFace import ArcMarginModel


def get_model(net_name):
    if net_name == "resnet101":
        return resnet101()
    elif net_name == "resnet50":
        return resnet50()
    elif net_name == "resnet18":
        return resnet18()
    elif net_name == "resnet34":
        return resnet34()
    elif net_name == "resnet_face18":
        return resnet_face18()
