import numpy as np
import random
import os
import json

from transformers import AutoModel, AutoTokenizer, AutoConfig


class Config(object):
    def __init__(self, args):
        self.args = args

        # train hyper parameter
        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.model_name = args.model_name

        # dataset
        self.dataset = args.dataset
        self.dataset_lang = args.dataset_lang

        # path and name
        self.root = args.root_path
        self.project_dir = os.path.join(self.root, args.project_dir)
        self.data_path = os.path.join(args.root_path, 'data', self.dataset)
        self.bert_path = args.bert_cache_dir

        # autoconfig BERT
        self.bert_type = args.bert_type
        bert_cfg = AutoConfig.from_pretrained(self.bert_type, cache_dir=self.bert_path)
        self.auto_model = AutoModel.from_pretrained(self.bert_type, config=bert_cfg, cache_dir=self.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type, cache_dir=self.bert_path)

        self.checkpoint_dir = self.project_dir + '/checkpoint/' + self.dataset
        self.log_dir = self.project_dir + '/log/' + self.dataset
        self.result_dir = self.project_dir + '/result/' + self.dataset
        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix
        self.test_prefix = args.test_prefix
#         self.add_scheduler = args.add_scheduler

        # relation num are read from file

        with open(os.path.join(self.data_path, "rel2id.json"), 'r', encoding='utf8') as wf:
            rel_num = json.load(wf)
            self.rel_num = len(rel_num[0])
            print(len(rel_num[0]))

        self.head_only = args.head_only
        # self.loop_type = args.loop_type
        self.step_dim = args.step_dim
        # self.method = args.method
        self.run_name = args.run_name
        self.model_save_name = args.model_name +"diff_name_" + args.run_name+ '_DATASET_N_' + self.dataset + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "_head_only_" + str(self.head_only)
        self.log_save_name = 'LOG_' + args.model_name+"diff_name_" + args.run_name + '_DATASET_N_' + self.dataset + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "_head_only_" + str(self.head_only)
        self.result_save_name = 'RESULT_' + args.model_name+"diff_name_" + args.run_name + '_DATASET_N_' + self.dataset + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "_head_only_" + str(self.head_only)+ ".json"

        random.seed(0)
        self.step_matrix = np.random.random((2 * self.max_len, self.step_dim))

        # log setting
        self.period = args.period
        self.test_epoch = args.test_epoch

        # debug
        self.debug = args.debug
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix

