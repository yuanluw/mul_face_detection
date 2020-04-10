# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/9 14:07, matt '


import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_w = 112
image_h = 112
channel = 3

num_workers = 4
print_freq = 100
grad_clip = 5.0

num_classes = 16520
data_path = "/media/wyl/mul_face_detection/train_data"

emb_size = 512
easy_margin = False
margin_m = 0.5
margin_s = 64.0



