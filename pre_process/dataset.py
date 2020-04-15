# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/5 21:43, matt '

# 该文件实现pytorch的数据读取接口

import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T


class Dataset(data.Dataset):
    def __init__(self, root, index="train", input_shape=(3, 112, 112)):
        self.input_shape = input_shape
        self.index = index
        imgs = []
        if index == "test":
            self.imgs = [root]
        else:
            if index == "train":
                data_path = os.path.join(root, "train_data", "train")
            else:
                data_path = os.path.join(root, "train_data", "val")

            for i in os.listdir(data_path):
                imgs.append(i)
            imgs = [os.path.join(data_path, img) for img in imgs]

            # 随机打乱数据
            self.imgs = np.random.permutation(imgs)
            print("read %d images" % (len(self.imgs)))
        normalize = T.Normalize(mean=[0.5], std=[0.5])
        if index == "train" or index == "val":
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, item):
        sample = self.imgs[item]
        if self.index == "train" or self.index == "val":
            img = Image.open(sample).convert('L')
            img = self.transforms(img)
            splits = sample.split(os.sep)[-1]
            label = int(splits.split('_')[1])
            label = torch.from_numpy(np.array(label))
            return img, label
        else:
            img = Image.open(sample).convert('L')
            img = self.transforms(img)
            return data

    def __len__(self):
        return len(self.imgs)


def get_dataset(arg, root, input_shape=(3, 112, 112), index="train"):
    dataset = Dataset(root=root, index=index, input_shape=input_shape)
    train_loader = data.DataLoader(dataset, batch_size=arg.train_batch_size)
    return train_loader


if __name__ == "__main__":
    root = "/media/wyl/mul_face_detection/train_data"
    dataset = Dataset(root=root, index="train", input_shape=(3, 112, 112))
    train_loader = data.DataLoader(dataset, batch_size=64)
    for i, (data, label) in enumerate(train_loader):
        print(label.shape)
