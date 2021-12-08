# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 上午10:41
# @Author  : godwaitup
# @FileName: data_prepare4debug.py

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='/home/zhaojiapeng/data', help='root path of dataset')
parser.add_argument('--dataset', type=str, default='CMED', choices=['NYT', 'WebNLG', 'CMED'], help='name of dataset')
parser.add_argument('--save_percent', type=float, default=0.2, help='data percentage to save')
parser.add_argument('--dataset_type', type=str, default='ALL', choices=['train', 'dev', 'test'], help='which type to choice')
args = parser.parse_args()


def choose_data():
    input_dataset = os.path.join(args.root_path, args.dataset)



    return


if __name__ == '__main__':

    choose_data()
