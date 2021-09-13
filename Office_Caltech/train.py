# -*- coding: utf-8 -*-
# @Time : 2021/9/11 10:20
# @Author : CHT
# @Site : 
# @File : train.py
# @Software: PyCharm 
# @Blog: https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Function:


import csv
from MADA import networks, train
from train_functions import get_data, data_path
import os
import torch

torch.cuda.set_device(1)
root_path = data_path.Office_Caltech_root_path
feas_dim = 4096
n_classes = 10
batch_size = 256
encoder_out_dim = 512

def train_to_csv(ds, dt, iterations=5, fea_type='DeCAF6', n_Dtl=0.03):
    domain_name = '{}_{}'.format(ds[:-4], dt[:-4])
    result_path = r'F:\Python_project\Experimental_Result\Office_Caltech\MADA\911_0.03'
    os.makedirs(result_path, exist_ok=True)

    # # Get features and labels
    # feas_ds, labels_ds = get_data.get_feas_labels(root_path, ds, fea_type=fea_type)
    # feas_dt, labels_dt = get_data.get_feas_labels(root_path, dt, fea_type=fea_type)
    # # Get dataloaders
    # dataloader_dt = get_data.get_src_dataloader_by_feas_labels(
    #     feas_dt, labels_dt, batch_size=batch_size, normalization=True, fea_type=fea_type)
    # dataloader_ds = get_data.get_src_dataloader_by_feas_labels(
    #     feas_ds, labels_ds, batch_size=batch_size, normalization=True, fea_type=fea_type)

    dataloader_ds, dataloader_dt = get_data.get_sd_td_with_labels_dataloader(
        root_path, ds, dt, fea_type=fea_type, n_Dtl=n_Dtl, batch_size=batch_size)
    print('{}, Ds:{}, Dt:{}'.format(domain_name, len(dataloader_ds.dataset), len(dataloader_dt.dataset)))

    train_epoch = 600
    domain_labels = [0.0, 1.0]
    loss_weights = [1, 0.5]
    dataloaders = [dataloader_ds, dataloader_dt]

    for _ in range(iterations):

        mada_net = networks.MADA_networks(in_dim=feas_dim, n_classes=n_classes, encoder_out_dim=encoder_out_dim).cuda()

        acc_tgt_best = train.training(
            dataloaders, mada_net, train_epoch, loss_weights, n_classes, lr=1e-2)

        with open(r'{}\{}.csv'.format(result_path, domain_name), 'a+',
                  newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow([float(acc_tgt_best)])




if __name__ == '__main__':

    train_to_csv(data_path.amazon_path, data_path.caltech_path)
    train_to_csv(data_path.amazon_path, data_path.dslr_path)
    train_to_csv(data_path.amazon_path, data_path.webcam_path)

    train_to_csv(data_path.caltech_path, data_path.amazon_path)
    train_to_csv(data_path.caltech_path, data_path.dslr_path)
    train_to_csv(data_path.caltech_path, data_path.webcam_path)
    #
    train_to_csv(data_path.dslr_path, data_path.amazon_path)
    train_to_csv(data_path.dslr_path, data_path.caltech_path)
    train_to_csv(data_path.dslr_path, data_path.webcam_path)

    train_to_csv(data_path.webcam_path, data_path.amazon_path)
    train_to_csv(data_path.webcam_path, data_path.caltech_path)
    train_to_csv(data_path.webcam_path, data_path.dslr_path)