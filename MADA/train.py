# -*- coding: utf-8 -*-
# @Time : 2021/9/11 9:55
# @Author : CHT
# @Site : 
# @File : train.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:

import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from MADA import networks
from train_functions.Optimizer_functions import *


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



def evaluate(net, dataloader_ds, dataloader_dt, alpha):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    net.eval()

    # init loss and accuracy
    loss_src = loss_tgt = 0
    acc_src = acc_tgt = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in dataloader_ds:
        images = Variable(images.type(FloatTensor)).reshape(images.shape[0], -1)
        labels = Variable(labels.type(LongTensor))

        # preds = classifier(encoder(images, labels))
        preds, _ = net(x=images, alpha=alpha)
        # loss += criterion(preds, labels).data[0]
        loss_src += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc_src += pred_cls.eq(labels.data).cpu().sum()

    loss_src /= len(dataloader_ds)
    acc_src = int(acc_src) / len(dataloader_ds.dataset)

    for (images, labels) in dataloader_dt:
        images = Variable(images.type(FloatTensor)).reshape(images.shape[0], -1)
        labels = Variable(labels.type(LongTensor))

        # preds = classifier(encoder(images, labels))
        preds, _ = net(images, alpha)
        # loss += criterion(preds, labels).data[0]
        loss_tgt += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc_tgt += pred_cls.eq(labels.data).cpu().sum()

    loss_tgt /= len(dataloader_dt)
    acc_tgt = int(acc_tgt) / len(dataloader_dt.dataset)

    print("Avg Loss = src:{:.2f}, tgt:{:.2f}, Avg Accuracy = src: {:2%} tgt:{:2%}".format(
        loss_src, loss_tgt,  acc_src, acc_tgt))
    return acc_src, acc_tgt


def configure_optimizers(net, n_classes, op='SGD'):
    # optimizer schedule 这里非常重要，学习了的设置每一个模块不一样
    # 当然主要更网络结构、参数量还有关系
    # 可以试试不同参数会导致什么结果，是overfitting还是underfitting
    model_parameter = [
        {
            "params": net.encoder.parameters(),
            "lr_mult": 0.1,
            'decay_mult': 2,
        },
        {
            "params": net.clf.parameters(),
            "lr_mult": 1.0,
            'decay_mult': 2,
        },
        *[
            {
            "params": net.domain_classifier_multi[class_idx].parameters(),
            "lr_mult":  1.0,
            'decay_mult': 2,
            } for class_idx in range(n_classes)
        ]
    ]

    if op == "SGD":
        optimizer = torch.optim.SGD(
            model_parameter,
            lr=0.01,
            momentum=0.9,
            weight_decay=2.5e-5,
            nesterov=True)

    elif op == "Adam":
        optimizer = torch.optim.Adam(
            model_parameter,
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=2.5e-5)
    else:
         raise Exception("Optimizer not implemented yet, please use Adam or SGD.")
    return optimizer


def get_lambda_p(p):
    return  2. / (1. + np.exp(10 * p)) - 1


def lr_schedule_step(optimizers, p):
    for param_group in optimizers.param_groups:
        # Update Learning rate
        param_group["lr"] = 1 * 0.01 / (
                    1 + 10 * p) ** 0.75
        # Update weight_decay
        param_group["weight_decay"] = 2.5e-5 * 2



def training(dataloaders, net, train_epoch, loss_weights, n_classes, lr=1e-3):
    criterion_d = torch.nn.BCEWithLogitsLoss()
    criterion_c = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(net.parameters(), lr=lr,)
    # optimizer schedule很重要
    optimizer = configure_optimizers(net, n_classes)
    iter_ds = iter_dt = acc_tgt_best = 0

    for epoch in range(train_epoch):
        p = float(1 + epoch * len(dataloaders[1])) / train_epoch / len(dataloaders[1])
        lr_schedule_step(optimizer, p)
        net.train()
        if epoch % len(dataloaders[0]) == 0:
            iter_ds = iter(dataloaders[0])
        if epoch % len(dataloaders[1]) == 0:
            iter_dt = iter(dataloaders[1])

        feas_ds, labels_ds = iter_ds.__next__()
        feas_dt, labels_dt = iter_dt.__next__()
        if feas_ds.shape[0] == 1 or feas_dt.shape[0] == 1:
            continue

        feas_ds = feas_ds.type(FloatTensor).reshape(feas_ds.shape[0], -1)
        labels_ds = labels_ds.type(LongTensor)
        feas_dt = feas_dt.type(FloatTensor).reshape(feas_dt.shape[0], -1)
        domain_ds = FloatTensor(feas_ds.size(0), 1).fill_(0)
        domain_dt = FloatTensor(feas_dt.size(0), 1).fill_(1)

        # -----------Ds
        c_output_ds, d_output_multi_dt = net(feas_ds)
        loss_c_ds = criterion_c(c_output_ds, labels_ds)
        loss_d_ds_multi = [
            criterion_d(d_output_multi_dt[class_inx], domain_ds)
            for class_inx in range(n_classes)
        ]

        # -----------Dt
        _, d_output_multi_dt = net(feas_dt)
        loss_d_dt_multi = [
            criterion_d(d_output_multi_dt[class_inx], domain_dt)
            for class_inx in range(n_classes)
        ]

        with OptimizerManager([optimizer]):
            loss_d = sum(loss_d_ds_multi) + sum(loss_d_dt_multi) / n_classes
            loss = loss_weights[0] * loss_c_ds + loss_weights[1] * loss_d
            loss.backward()

        if epoch % 30 == 0:
            acc_src, acc_tgt = evaluate(net, dataloaders[0], dataloaders[1], alpha=0.5)
            acc_tgt_best = acc_tgt if acc_tgt_best < acc_tgt else acc_tgt_best
    return acc_tgt_best