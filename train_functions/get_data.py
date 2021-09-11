# -*- coding: utf-8 -*-
# @Time : 2021/3/13 14:56
# @Author : CHT
# @Site : 
# @File : get_data.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:


import numpy as np
import os
import torch.utils.data as data
from torchvision import datasets, transforms
import torch
from sklearn.cluster import SpectralClustering, KMeans
import collections
from scipy.io import loadmat
from train_functions import normalization_data


def separate_data(feas, labels, index, in_index=True):
    # 根据索引分离数据
    if in_index:
        feas = np.asarray([feas[i] for i in index])
        labels = np.asarray([labels[i] for i in index])

    else:
        feas = np.asarray([feas[i] for i in range(len(feas)) if i not in index])
        labels = np.asarray([labels[i] for i in range(len(labels)) if i not in index])
    return feas, labels


def concatenate_data(feas, labels, feas_extra, labels_extra):
    if len(np.asarray(feas).shape) == len(np.asarray(feas_extra).shape):
        feas = np.concatenate((feas, feas_extra), 0)
        labels = np.concatenate((labels, labels_extra), 0)
    return  feas, labels


def list_numpy(feas_list, labels_list):
    # 保存的时候是list，使用的时候就转为numpy
    feas = feas_list[0]
    labels = labels_list[0]
    for i in range(1, len(feas_list)):
        feas = np.concatenate((feas, feas_list[i]), 0)
        labels = np.concatenate((labels, labels_list[i]), 0)
    return feas, labels


def get_feas_labels(root_path, domain, fea_type='Resnet50'):
    # 得到原始特征
    path = os.path.join(root_path, domain)
    if fea_type == 'Resnet50':
        with open(path, encoding='utf-8') as f:
            imgs_data = np.loadtxt(f, delimiter=",")
            features = imgs_data[:, :-1]
            labels = imgs_data[:, -1]

    elif fea_type == 'MDS':
        # dict_keys(['__header__', '__version__', '__globals__', 'fts', 'labels'])
        domain_data = loadmat(path)
        features = np.asarray(domain_data['fts'])
        labels = np.asarray(domain_data['labels']).squeeze()

    else: # DeCAF6
        domain_data = loadmat(path)
        features = np.asarray(domain_data['feas'])
        labels = np.asarray(domain_data['labels']).squeeze() - 1  # start from 0
    return features, labels


def get_src_dataloader_by_feas_labels(feas, labels, batch_size=128, drop_last=False,
                                      normalization=False, fea_type='Resnet50'):
    # get dataloader
    if normalization:
        dataset = normalization_data.myDataset(feas, labels, fea_type)
    else:
        dataset = data.TensorDataset(torch.tensor(feas), torch.tensor(labels))
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    return dataloader


def get_ds_dtl_dtu(feas_ds, labels_ds, feas_dt, labels_dt, n_dtl):
    extra_index = np.random.choice(len(feas_dt), int(n_dtl * len(feas_dt)), replace=False)

    fea_tgt_label = np.asarray([feas_dt[i] for i in extra_index])
    labels_tgt_label = np.asarray([labels_dt[i] for i in extra_index])

    feas_dt = np.asarray([feas_dt[i] for i in range(len(feas_dt)) if i not in extra_index])
    labels_dt = np.asarray([labels_dt[i] for i in range(len(labels_dt)) if i not in extra_index])

    feas_ds = np.concatenate((feas_ds, fea_tgt_label), 0)
    labels_ds = np.concatenate((labels_ds, labels_tgt_label), 0)

    return feas_ds, labels_ds, feas_dt, labels_dt


def get_sd_td_with_labels_dataloader(root_path, ds, dt, n_Dtl, fea_type, batch_size=100):
    feas_src, labels_src = get_feas_labels(root_path, ds, fea_type=fea_type)
    feas_tgt, labels_tgt = get_feas_labels(root_path, dt, fea_type=fea_type)

    feas_src, labels_src, feas_tgt, labels_tgt = get_ds_dtl_dtu(feas_src, labels_src, feas_tgt, labels_tgt, n_Dtl)

    fea_type = data.TensorDataset(torch.tensor(feas_src), torch.tensor(labels_src))
    dataloader_src = data.DataLoader(
        dataset=fea_type,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    fea_type = data.TensorDataset(torch.tensor(feas_tgt), torch.tensor(labels_tgt))
    dataloader_tgt = data.DataLoader(
        dataset=fea_type,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    return dataloader_src, dataloader_tgt