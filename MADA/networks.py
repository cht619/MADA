# -*- coding: utf-8 -*-
# @Time : 2021/9/11 9:00
# @Author : CHT
# @Site : 
# @File : networks.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha

        # Must return same number as inputs to forward()
        return output, None


class Endocer(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super(Endocer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(Classifier, self).__init__()
        self.clf = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.clf(x)


class Domain_classifier(nn.Module):
    def __init__(self, in_dim):
        super(Domain_classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


class MADA_networks(nn.Module):
    def __init__(self, in_dim, encoder_out_dim, n_classes):
        super(MADA_networks, self).__init__()
        self.n_classes = n_classes
        self.encoder = Endocer(in_dim)
        self.clf = Classifier(in_dim=encoder_out_dim, n_classes=n_classes)
        self.domain_classifier_multi = [
            Domain_classifier(encoder_out_dim).cuda() for _ in range(n_classes)
        ]

    def forward(self, x, alpha=0.9):
        x = self.encoder(x)
        c_output = self.clf(x)

        reversal_feas = GRL.apply(x, alpha)
        d_output_multi = []
        for class_inx in range(self.n_classes):
            weighted_reversal_feas = F.softmax(c_output, dim=1)[:, class_inx].unsqueeze(1) * reversal_feas
            d_output_multi.append(
                self.domain_classifier_multi[class_inx](weighted_reversal_feas)
            )
        return c_output, d_output_multi


