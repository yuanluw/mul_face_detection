# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/5 21:09, matt '

# 该文件用于对测试集文件重新整理


import os
import shutil
import random

root_path = "F:\\python\\mul_face detection"

if __name__ == "__main__":
    if not os.path.exists(os.path.join(root_path, "train_data")):
        os.makedirs(os.path.join(root_path, "train_data"))

    if not os.path.exists(os.path.join(root_path, "train_data", "train")):
        os.makedirs(os.path.join(root_path, "train_data", "train"))

    if not os.path.exists(os.path.join(root_path, "train_data", "val")):
        os.makedirs(os.path.join(root_path, "train_data", "val"))

    race_list = ['African', 'Caucasian', "Asian", 'Indian']
    id = 0
    index = 0
    for race in race_list:
        race_count_list = []
        data_path = os.path.join(root_path, "training", race)
        data_list = os.listdir(data_path)
        data_len = len(data_list)

        if race == 'Caucasian':
            val_num = int(data_len * 0.4)
        else:
            val_num = int(data_len * 0.1)

        val_list = random.sample(data_list, val_num)
        train_list = list(set(data_list).difference(set(val_list)))

        print(race, " have: ", data_len)
        for i in train_list:
            pic_file = os.path.join(data_path, i)
            race_count_list.append(len(os.listdir(pic_file)))
            for ind, j in enumerate(os.listdir(pic_file)):
                pic = os.path.join(pic_file, j)
                new_pic = os.path.join(root_path, "train_data", "train", str(index) + "_" + str(id)+"_"+str(ind)+".jpg")
                shutil.copyfile(pic, new_pic)
            id += 1
        for i in val_list:
            pic_file = os.path.join(data_path, i)
            race_count_list.append(len(os.listdir(pic_file)))
            for ind, j in enumerate(os.listdir(pic_file)):
                pic = os.path.join(pic_file, j)
                new_pic = os.path.join(root_path, "train_data", "val", str(index) + "_" + str(id)+"_"+str(ind)+".jpg")
                shutil.copyfile(pic, new_pic)
            id += 1

        index += 1