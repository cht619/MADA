# -*- coding: utf-8 -*-
# @Time : 2021/3/13 14:55
# @Author : CHT
# @Site : 
# @File : normalization_data.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function: 对数据标准化

import torch.utils.data as data
from torchvision import datasets, transforms
import os
import numpy as np


def get_path(root_path, images_domain):
    # 多个类别总结在一起，返回一个list
    images_path = []
    labels = []
    domain_classes_path = os.path.join(os.path.join(root_path, images_domain))
    domain_classes = os.listdir(domain_classes_path)
    domain_classes.sort(key=lambda x: int(x))

    for c in domain_classes:
        # 获取每一个类别的的路径。
        imgs_path = os.path.join(domain_classes_path, c)
        # 获取每一张图片的路径
        for img in os.listdir(imgs_path):
            img_path = os.path.join(domain_classes_path, c, img)
            images_path.append(img_path)
            labels.append(int(c))
    return images_path, labels


class myDataset(data.Dataset):
    def __init__(self, imgs_data, labels, feature_type):
        self.imgs_data = imgs_data
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5],[0.5])])
        if feature_type == 'DeCAF6':
            self.feature = 4096
        elif feature_type == 'Resnet50':
            self.feature = 2048
        elif feature_type == 'MDS':
            self.feature = 400

    def __getitem__(self, index):
        imgs_data = np.asarray(self.imgs_data[index])
        # imgs_data = imgs_data[:, np.newaxis]
        # 这里不知道为什么不加多一维，会报错，说输入只能是2或者3维
        # 加多一维 却变成[1,800,1] 太奇怪了，只能reshape
        # imgs_data = self.transform(Image.fromarray(imgs_data[:, np.newaxis])).reshape(1, 4096)
        imgs_data = self.transform(imgs_data[:, np.newaxis]).reshape(1, self.feature)
        return imgs_data, self.labels[index]

    # 可以直接打开路径
    # def __getitem__(self, index):
    #     img_data = self.transform(Image.open(self.imgs_path[index]))
    #     if img_data.shape[0] == 1:
    #         img_data = img_data.expand(3, img_data.shape[1], img_data.shape[2])
    #     return img_data, self.labels[index]

    def __len__(self):
        return len(self.imgs_data)