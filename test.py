# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/15 17:25, matt '


import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from time import time
import scipy.io as sio
from scipy.spatial.distance import pdist

import torch
import torchvision.transforms as T

import config
from model import resnet_face18


def cosin_metric(x1, x2):
    return np.dot(x1, x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))


def get_image(path, input_shape):
    normalize = T.Normalize(mean=[0.5], std=[0.5])

    img = Image.open(path).convert('L')
    transforms = T.Compose([
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        normalize
    ])
    img = transforms(img)
    img = img.unsqueeze(0)
    return img


def get_features(model, test_list):
    process_bar = tqdm(total=len(test_list))
    for idx, img_path in enumerate(test_list):
        process_bar.update(1)

        img = get_image(img_path, (config.channel, config.image_h, config.image_w))
        img = img.to(config.device)
        if idx == 0:
            feature = model(img)
            feature = feature.detach().cpu().numpy()
            features = feature
        else:
            feature = model(img)
            feature = feature.detach().cpu().numpy()
            features = np.concatenate((features, feature), axis=0)

    return features


def get_feature_dict(arg):
    checkpoint = torch.load(os.path.join(config.checkpoint_path, arg.checkpoint))
    model = checkpoint['feature_net'].module.to(config.device)

    model.eval()
    name_list = [name for name in os.listdir(config.test_path)]
    img_paths = [os.path.join(config.test_path, name) for name in name_list]
    print("test image number: ", len(img_paths))
    get_feature_time = time()
    features = get_features(model, img_paths)
    get_feature_time = time() - get_feature_time
    print(features.shape)
    print('total time is {}, avg time is {}'.format(get_feature_time, get_feature_time / len(img_paths)))

    fe_dict = {}
    for i, each in enumerate(name_list):
        fe_dict[each] = features[i]
    sio.savemat('face_embedding_test.mat', fe_dict)


def calculate_similarity():
    face_features = sio.loadmat('face_embedding_test.mat')
    sample_sub = open('submission_template.csv', 'r')
    sub = open('submission_new.csv', 'w')
    print("loader csv")
    print(len(face_features))
    lines = sample_sub.readlines()
    process_bar = tqdm(total=len(lines))
    for line in lines:
        process_bar.update(1)
        pair = line.split(',')[0]
        sub.write(pair + ',')
        a, b = pair.split(":")
        score = cosin_metric(face_features[a][0], face_features[b][0])
        score = (score + 0.3) / 1.3
        score = "%.5f" % score
        sub.write(score + '\n')
    sample_sub.close()
    sub.close()







