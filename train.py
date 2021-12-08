import os
import torch
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import config
from framework import framework

import argparse
import models
import os
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='SGCN', help='name of the model')

parser.add_argument('--root_path', type=str, default='/home/zhaojiapeng/', help='name of the model.')
parser.add_argument('--project_dir', type=str, default='SEPC', help='project dir name.')
parser.add_argument('--dataset_path', type=str, default='/home/zhaojiapeng/data/', help='path of the dataset.')
parser.add_argument('--bert_cache_dir', type=str, default='/home/zhaojiapeng/data/bert_cache', help='bert_cache_path.')
parser.add_argument('--bert_type', type=str, default='bert-base-cased', choices=['bert-base-cased', 'hfl/chinese-bert-wwm'], help='bert type of hungging face.')


parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='NYT')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_prefix', type=str, default='train_triples')
parser.add_argument('--dev_prefix', type=str, default='dev_triples')
parser.add_argument('--test_prefix', type=str, default='test_triples')
parser.add_argument('--max_len', type=int, default=200)
parser.add_argument('--period', type=int, default=1000)
parser.add_argument('--step_dim', type=int, default=12)
parser.add_argument('--debug', type=bool, default=False)


parser.add_argument('--dataset_lang', type=str, choices=['CHINESE', 'ENGLISH'], default='ENGLISH')
parser.add_argument('--head_only', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--run_name', type=str, default='SEPC')
parser.add_argument('--add_wandb', type=bool, default=False)

args = parser.parse_args()

seed = args.seed
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

logger_wandb = None
if args.add_wandb:
    wandb.init(project=args.run_name, name="{}_{}".format(args.model_name, args.dataset), config=args)
    logger_wandb = wandb

con = config.Config(args)


current_dir = os.path.abspath(os.getcwd())
print(current_dir)

fw = framework.Framework(con, logger_wandb)

model = {
    'SGCN': models.SGCN,
    'Casrel': models.Casrel,
}

fw.train(model[args.model_name])