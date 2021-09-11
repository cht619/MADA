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

root_path = data_path.Office_Home_root_path
feas_dim = 2048
n_classes = 65
batch_size = 256
encoder_out_dim = 512

def train_to_csv(ds, dt, iterations=5, fea_type='Resnet50', n_Dtl=0.03):
    domain_name = '{}_{}'.format(ds[:-4], dt[:-4])
    result_path = r'F:\Python_project\Experimental_Result\Office_Home\MADA\911_0.03'
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
    train_to_csv(data_path.domain_ar, data_path.domain_ar_cl)
    train_to_csv(data_path.domain_ar, data_path.domain_ar_pr)
    train_to_csv(data_path.domain_ar, data_path.domain_ar_rw)

    train_to_csv(data_path.domain_cl, data_path.domain_cl_ar)
    train_to_csv(data_path.domain_cl, data_path.domain_cl_pr)
    train_to_csv(data_path.domain_cl, data_path.domain_cl_rw)

    train_to_csv(data_path.domain_pr, data_path.domain_pr_ar)
    train_to_csv(data_path.domain_pr, data_path.domain_pr_cl)
    train_to_csv(data_path.domain_pr, data_path.domain_pr_rw)

    train_to_csv(data_path.domain_rw, data_path.domain_rw_ar)
    train_to_csv(data_path.domain_rw, data_path.domain_rw_cl)
    train_to_csv(data_path.domain_rw, data_path.domain_rw_pr)