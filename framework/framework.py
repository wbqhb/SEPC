# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 下午3:05
# @Author  : godwaitup
# @FileName: framework.py
# original framework for joint extraction.

import torch.optim as optim
from torch import nn
import os
import data_loader
import torch.nn.functional as F
import numpy as np
import json
from functools import partial
from data_loader import cmed_collate_fn
import torch

def _to_sub(triple_list, head_only=False, lang='ENGLISH'):
    ret = set()
    for triple in triple_list:
        if lang is 'CHINESE':
            triple = (triple[0].replace('$', ' ').lower(), triple[1], triple[2].replace('$', ' ').lower())

        if head_only:
            ret.add(triple[0].split(" ")[0])
        else:
            ret.add(triple[0])
    return ret


def _to_obj(triple_list, head_only=False, lang='ENGLISH'):
    ret = set()
    for triple in triple_list:
        if lang is 'CHINESE':
            triple = (triple[0].replace('$', ' ').lower(), triple[1], triple[2].replace('$', ' ').lower())

        if head_only:
            ret.add(triple[2].split(" ")[0])
        else:
            ret.add(triple[2])
    return ret


def _to_ep(triple_list, head_only=False, lang='ENGLISH'):
    ret = set()
    for triple in triple_list:
        if lang is 'CHINESE':
            triple = (triple[0].replace('$', ' ').lower(), triple[1], triple[2].replace('$', ' ').lower())

        if head_only:
            _h = triple[0].split(" ")
            _t = triple[2].split(" ")
            ret.add(tuple((_h[0], _t[0])))
        else:
            ret.add(tuple((triple[0], triple[2])))
    return ret


def _to_triple(triple_list, head_only=False, lang='ENGLISH'):
    ret = set()
    for triple in triple_list:

        # print("lang:{} A:{}".format(lang, triple))
        if lang is 'CHINESE':
            triple = (triple[0].replace('$', ' ').lower(), triple[1], triple[2].replace('$', ' ').lower())
        # print("B:{}".format(triple))


        if head_only:
            _h = triple[0].split(" ")
            _t = triple[2].split(" ")
            ret.add(tuple((_h[0], triple[1], _t[0])))
        else:
            ret.add(tuple((triple[0], triple[1], triple[2])))
    return ret


def _load_gold_data(data_gold, data_id, head_only=False, gold_type='EP', lang='ENGLISH'):
    _tokens, _triples = data_gold[data_id]
    if gold_type == 'EP':
        gold_value = _to_ep(_triples, head_only, lang=lang)
    elif gold_type == 'sub':
        gold_value = _to_sub(_triples, head_only, lang=lang)
    elif gold_type == 'obj':
        gold_value = _to_obj(_triples, head_only, lang=lang)
    elif gold_type == 'ALL':
        gold_value = _to_triple(_triples, head_only, lang=lang)

    return gold_value, _tokens


def _cal_prf(correct_num, predict_num, gold_num):
    eval_p = correct_num / (predict_num + 1e-10)
    eval_r = correct_num / (gold_num + 1e-10)
    eval_f = 2 * eval_p * eval_r / (eval_p + eval_r + 1e-10)
    return eval_p, eval_r, eval_f


class Framework(object):
    def __init__(self, con, wandb_log):
        self.config = con
        self.wandb_log = wandb_log

    def train(self, model_pattern):
        # initialize the model
        ori_model = model_pattern(self.config)
        ori_model.cuda()

        # define the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ori_model.parameters()), lr=self.config.learning_rate)

        # whether use multi GPU
        if self.config.multi_gpu:
            model = nn.DataParallel(ori_model)
        else:
            model = ori_model

        # define the loss function
        def loss(gold, pred, mask):
            pred = pred.squeeze(-1)
            los = F.binary_cross_entropy(pred, gold, reduction='none')

            if mask is None:
                los = torch.sum(los)/self.config.rel_num
                return los

            if los.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            los = torch.sum(los * mask) / torch.sum(mask)
            return los

        # check the checkpoint dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # get the data loader
        train_data_loader = data_loader.get_loader(self.config, tokenizer=self.config.tokenizer, prefix=self.config.train_prefix, collate_fn=partial(cmed_collate_fn, num_rels=self.config.rel_num))
        dev_data_loader = data_loader.get_loader(self.config, tokenizer=self.config.tokenizer, prefix=self.config.dev_prefix, is_test=True, collate_fn=partial(cmed_collate_fn, num_rels=self.config.rel_num))
        test_data_loader = data_loader.get_loader(self.config, tokenizer=self.config.tokenizer,
                                                 prefix=self.config.test_prefix, is_test=True,
                                                 collate_fn=partial(cmed_collate_fn, num_rels=self.config.rel_num))


        model.train()
        global_step = 0
        loss_sum = 0
        ent_boundary_loss_sum = 0
        ent_span_loss_sum = 0
        ent_pair_loss_sum = 0
        rel_loss_sum = 0
        best_f1_score = -1
        best_test_f1 = 0
        best_test_h_f1 = 0

        best_epoch = 0

        # the training loop
        for epoch in range(self.config.max_epoch):
            train_data_prefetcher = data_loader.DataPreFetcher(train_data_loader)
            data = train_data_prefetcher.next()

            while data is not None:

                if self.config.model_name == 'SGCN' or self.config.model_name == 'SGCN_NO_STEP':
                    pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails, \
                    sim_sub_h2t, sim_sub_t2h, sim_obj_h2t, sim_obj_t2h, \
                    sim_sub_oh, sim_sub_ot, sim_obj_sh, sim_obj_st, pred_rels = model(data)

                    # entity boundary loss
                    loss_sub_heads = loss(data['em_sub_heads'], pred_sub_heads, mask=data['mask'])
                    loss_sub_tails = loss(data['em_sub_tails'], pred_sub_tails, mask=data['mask'])
                    loss_obj_heads = loss(data['em_obj_heads'], pred_obj_heads, mask=data['mask'])
                    loss_obj_tails = loss(data['em_obj_tails'], pred_obj_tails, mask=data['mask'])

                    # entity span loss
                    loss_sub_h2t = loss(data['sub_h2t'], sim_sub_h2t, mask=data['mask'])
                    loss_sub_t2h = loss(data['sub_t2h'], sim_sub_t2h, mask=data['mask'])
                    loss_obj_h2t = loss(data['obj_h2t'], sim_obj_h2t, mask=data['mask'])
                    loss_obj_t2h = loss(data['obj_t2h'], sim_obj_t2h, mask=data['mask'])

                    # entity pair loss
                    loss_sub2objh = loss(data['sub2obj_h'], sim_sub_oh, mask=data['mask'])
                    loss_sub2objt = loss(data['sub2obj_t'], sim_sub_ot, mask=data['mask'])
                    loss_obj2subh = loss(data['obj2sub_h'], sim_obj_sh, mask=data['mask'])
                    loss_obj2subt = loss(data['obj2sub_t'], sim_obj_st, mask=data['mask'])

                    # relation loss
                    loss_rel = loss(data['rel_labels'], pred_rels, mask=None)

                    ent_boundary_loss = loss_sub_heads + loss_sub_tails + loss_obj_heads + loss_obj_tails
                    ent_span_loss = loss_sub_h2t + loss_sub_t2h + loss_obj_h2t + loss_obj_t2h
                    ent_pair_loss = loss_sub2objh + loss_sub2objt + loss_obj2subh + loss_obj2subt

                    total_loss = ent_boundary_loss + ent_span_loss + ent_pair_loss + loss_rel

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    global_step += 1
                    loss_sum += total_loss.item()
                    ent_boundary_loss_sum += ent_boundary_loss.item()
                    ent_span_loss_sum += ent_span_loss.item()
                    ent_pair_loss_sum += ent_pair_loss.item()
                    rel_loss_sum += loss_rel.item()

                    if global_step % self.config.period == 0:
                        # print(loss_sum)
                        if self.wandb_log is not None:
                            self.wandb_log.log({"LOSS_SUM:": loss_sum})

                        loss_sum = 0
                        ent_boundary_loss_sum = 0
                        ent_span_loss_sum = 0
                        ent_pair_loss_sum = 0
                        rel_loss_sum = 0

                    data = train_data_prefetcher.next()

                elif self.config.model_name == 'Casrel':

                    pred_sub_heads, pred_sub_tails, pred_s2ro_heads, pred_s2ro_tails = model(data)

                    # entity boundary loss
                    loss_sub_heads = loss(data['em_sub_heads'], pred_sub_heads, mask=data['mask'])
                    loss_sub_tails = loss(data['em_sub_tails'], pred_sub_tails, mask=data['mask'])

                    # relation loss
                    loss_s2ro_heads = loss(data['batch_s2ro_heads'], pred_s2ro_heads, mask=data['mask'])
                    loss_s2ro_tails = loss(data['batch_s2ro_tails'], pred_s2ro_tails, mask=data['mask'])

                    ent_boundary_loss = loss_sub_heads + loss_sub_tails
                    rel_loss = loss_s2ro_heads + loss_s2ro_tails

                    total_loss = ent_boundary_loss + rel_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    global_step += 1
                    loss_sum += total_loss.item()
                    ent_boundary_loss_sum += ent_boundary_loss.item()
                    rel_loss_sum += rel_loss.item()

                    if global_step % self.config.period == 0:
                        if self.wandb_log is not None:
                            self.wandb_log.log({"LOSS_SUM:": loss_sum})

                        loss_sum = 0
                        ent_boundary_loss_sum = 0
                        ent_span_loss_sum = 0
                        ent_pair_loss_sum = 0
                        rel_loss_sum = 0

                    data = train_data_prefetcher.next()

            if (epoch + 1) % self.config.test_epoch == 0:
                model.eval()
                # call the test function
                dev_triple_p, dev_triple_r, dev_triple_f, dev_triple_hp, dev_triple_hr, dev_triple_hf, \
                dev_ep_p, dev_ep_r, dev_ep_f, dev_ep_hp, dev_ep_hr, dev_ep_hf, \
                dev_sub_p, dev_sub_r, dev_sub_f, dev_sub_hp, dev_sub_hr, dev_sub_hf, \
                dev_obj_p, dev_obj_r, dev_obj_f, dev_obj_hp, dev_obj_hr, dev_obj_hf = self.test(dev_data_loader, self.config.step_dim, self.config.step_matrix, model)

                test_triple_p, test_triple_r, test_triple_f, test_triple_hp, test_triple_hr, test_triple_hf, \
                test_ep_p, test_ep_r, test_ep_f, test_ep_hp, test_ep_hr, test_ep_hf, \
                test_sub_p, test_sub_r, test_sub_f, test_sub_hp, test_sub_hr, test_sub_hf, \
                test_obj_p, test_obj_r, test_obj_f, test_obj_hp, test_obj_hr, test_obj_hf = self.test(test_data_loader,
                                                                                                self.config.step_dim,
                                                                                                self.config.step_matrix,
                                                                                                model)

                model.train()

                # eval_f1_score
                if dev_triple_f > best_f1_score:

                    best_epoch = epoch
                    best_f1_score = dev_triple_f
                    best_test_h_f1 = test_triple_hf
                    best_test_f1 = test_triple_f

                    # save the best model
                    path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
                    if not self.config.debug:
                        torch.save(ori_model.state_dict(), path)

                if self.wandb_log is not None:
                    self.wandb_log.log({
                        "BEST_EPOCH:": best_epoch,

                        "DEV_Triple_F1": dev_triple_f,
                        "DEV_TripleH_F1": dev_triple_hf,

                        "DEV_EP_F1": dev_ep_f,
                        "DEV_SUB_F1": dev_sub_f,
                        "DEV_OBJ_F1": dev_obj_f,

                        "DEV_EPH_F1": dev_ep_hf,
                        "DEV_SUBH_F1": dev_sub_hf,
                        "DEV_OBJH_F1": dev_obj_hf,

                        "best_test_h_f1": best_test_h_f1,
                        "best_test_f1": best_test_f1,

                        "current_epoch": epoch})

            # manually release the unused cache
            torch.cuda.empty_cache()

    def cal_sub_prob(self, head_idx, tail_idx, trans_head_idx, trans_tail_idx, pred_heads, pred_tails, head_walk_step, tail_walk_step, model, encoded_txt, seq_len):

        _head_prob = pred_heads[0][head_idx][0].tolist()
        _tail_prob = pred_tails[0][tail_idx][0].tolist()
        _head_mapping = torch.Tensor(1, 1, encoded_txt.size(1)).zero_()
        _tail_mapping = torch.Tensor(1, 1, encoded_txt.size(1)).zero_()
        _head_mapping[0][0][head_idx] = 1
        _tail_mapping[0][0][tail_idx] = 1
        _head_mapping = _head_mapping.to(encoded_txt)
        _tail_mapping = _tail_mapping.to(encoded_txt)

        sub_span = model.gen_span_emb(torch.LongTensor([head_idx]), torch.LongTensor([tail_idx]), encoded_txt)

        # predict entity span
        sim_ent_ht, sim_ent_th = model.sub_span_trans(_head_mapping, _tail_mapping, head_walk_step, tail_walk_step, encoded_txt, seq_len)
        _h2t_prob = sim_ent_ht[0][tail_idx][0].tolist()
        _t2h_prob = sim_ent_th[0][head_idx][0].tolist()
        # span_prob = _head_prob * _h2t_prob + _tail_prob * _t2h_prob

        # trans head idx
        sim_ent_gh, sim_ent_gt = model.sub_entity_trans(sub_span, head_walk_step, tail_walk_step, encoded_txt, seq_len)
        trans_head_prob = sim_ent_gh[0][trans_head_idx][0].tolist()
        trans_tail_prob = sim_ent_gt[0][trans_tail_idx][0].tolist()

        return _head_prob, _h2t_prob, _tail_prob, _t2h_prob, trans_head_prob, trans_tail_prob

    def cal_obj_prob(self, head_idx, tail_idx, trans_head_idx, trans_tail_idx, pred_heads, pred_tails, head_walk_step, tail_walk_step, model, encoded_txt, seq_len):

        _head_prob = pred_heads[0][head_idx][0].tolist()
        _tail_prob = pred_tails[0][tail_idx][0].tolist()
        _head_mapping = torch.Tensor(1, 1, encoded_txt.size(1)).zero_()
        _tail_mapping = torch.Tensor(1, 1, encoded_txt.size(1)).zero_()
        _head_mapping[0][0][head_idx] = 1
        _tail_mapping[0][0][tail_idx] = 1
        _head_mapping = _head_mapping.to(encoded_txt)
        _tail_mapping = _tail_mapping.to(encoded_txt)

        obj_span = model.gen_span_emb(torch.LongTensor([head_idx]), torch.LongTensor([tail_idx]), encoded_txt)
        # predict entity span
        sim_ent_ht, sim_ent_th = model.obj_span_trans(_head_mapping, _tail_mapping, head_walk_step, tail_walk_step, encoded_txt, seq_len)
        _h2t_prob = sim_ent_ht[0][tail_idx][0].tolist()
        _t2h_prob = sim_ent_th[0][head_idx][0].tolist()
        # span_prob = _head_prob * _h2t_prob + _tail_prob * _t2h_prob

        # trans head idx
        sim_ent_gh, sim_ent_gt = model.obj_entity_trans(obj_span, head_walk_step, tail_walk_step, encoded_txt, seq_len)
        trans_head_prob = sim_ent_gh[0][trans_head_idx][0].tolist()
        trans_tail_prob = sim_ent_gt[0][trans_tail_idx][0].tolist()


        return _head_prob, _h2t_prob, _tail_prob, _t2h_prob, trans_head_prob, trans_tail_prob

    def cal_rel_prob(self, sub_head_idx, sub_tail_idx, obj_head_idx, obj_tail_idx, model, encoded_txt, rel_bar=0.5):

        sub_head_mapping = torch.Tensor(1, 1, encoded_txt.size(1)).zero_()
        sub_tail_mapping = torch.Tensor(1, 1, encoded_txt.size(1)).zero_()
        sub_head_mapping[0][0][sub_head_idx] = 1
        sub_tail_mapping[0][0][sub_tail_idx] = 1
        sub_head_mapping = sub_head_mapping.to(encoded_txt)
        sub_tail_mapping = sub_tail_mapping.to(encoded_txt)

        obj_head_mapping = torch.Tensor(1, 1, encoded_txt.size(1)).zero_()
        obj_tail_mapping = torch.Tensor(1, 1, encoded_txt.size(1)).zero_()
        obj_head_mapping[0][0][obj_head_idx] = 1
        obj_tail_mapping[0][0][obj_tail_idx] = 1
        obj_head_mapping = obj_head_mapping.to(encoded_txt)
        obj_tail_mapping = obj_tail_mapping.to(encoded_txt)

        pred_rels = model.rel_classification(sub_head_mapping, sub_tail_mapping, obj_head_mapping, obj_tail_mapping, encoded_txt)
        pred_rels_idx = np.where(pred_rels.cpu()[0] > rel_bar)[0]
        return pred_rels_idx

    def _cal_ep_score(self, sub_span_prob, obj_span_prob, sub_trans_prob, obj_trans_prob):
        _score = sub_span_prob*sub_trans_prob + obj_span_prob*obj_trans_prob
        return _score

    def test(self, x_data_loader, step_dim, step_matrix, model):
        test_data_prefetcher = data_loader.DataPreFetcher(x_data_loader)
        data = test_data_prefetcher.next()
        pred_eps_id = list()
        data_id = 0
        data_gold = list()

        id2rel = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[0]
        print(id2rel)

        def make_step(sample_idx, text_len):
            walk_step = np.zeros((text_len, step_dim))
            for i in range(text_len):
                walk_step[i] = step_matrix[i - sample_idx + self.config.max_len]
            walk_step_t = torch.Tensor(walk_step)
            walk_step_t = walk_step_t.unsqueeze(0)
            walk_step_t = walk_step_t.to(torch.device('cuda'))
            return walk_step_t

        while data is not None:
            with torch.no_grad():
                token_ids = data['token_ids']
                tokens = data['tokens'][0]
                mask = data['mask']
                gold_triples = data['triples'][0]
                data_gold.append((tokens, gold_triples))
                seq_len = len(tokens)
                encoded_text = model.get_encoded_text(token_ids, mask)

                if self.config.model_name == 'SGCN' or self.config.model_name == 'SGCN_NO_STEP':
                    pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model.pred_entity_boundary(encoded_text)

                    _bar = 0.1
                    max_span_len = 30
                    span_sub_heads = np.where(pred_sub_heads.cpu()[0] > _bar)[0]
                    span_sub_tails = np.where(pred_sub_tails.cpu()[0] > _bar)[0]
                    span_obj_heads = np.where(pred_obj_heads.cpu()[0] > _bar)[0]
                    span_obj_tails = np.where(pred_obj_tails.cpu()[0] > _bar)[0]

                    pred_eps = dict()
                    for _sub_head_idx in span_sub_heads:
                        for _sub_tail_idx in span_sub_tails:
                            for _obj_head_idx in span_obj_heads:
                                for _obj_tail_idx in span_obj_tails:

                                    if _sub_head_idx <= _sub_tail_idx and _obj_head_idx <= _obj_tail_idx and (_sub_tail_idx - _sub_head_idx) < max_span_len and (_obj_tail_idx - _obj_head_idx) < max_span_len:

                                        sub_head_walk_step = make_step(_sub_head_idx, seq_len)
                                        sub_tail_walk_step = make_step(_sub_tail_idx, seq_len)
                                        obj_head_walk_step = make_step(_obj_head_idx, seq_len)
                                        obj_tail_walk_step = make_step(_obj_tail_idx, seq_len)

                                        # cal span prob and trans prob
                                        sub_head_prob, sub_h2t_prob, sub_tail_prob, sub_t2h_prob, sub_trans_head_prob, sub_trans_tail_prob = \
                                            self.cal_sub_prob(_sub_head_idx, _sub_tail_idx, _obj_head_idx, _obj_tail_idx, pred_sub_heads, pred_sub_tails, sub_head_walk_step, sub_tail_walk_step, model, encoded_text, seq_len)

                                        obj_head_prob, obj_h2t_prob, obj_tail_prob, obj_t2h_prob, obj_trans_head_prob, obj_trans_tail_prob = \
                                            self.cal_obj_prob(_obj_head_idx, _obj_tail_idx, _sub_head_idx, _sub_tail_idx, pred_obj_heads, pred_obj_tails, obj_head_walk_step, obj_tail_walk_step, model, encoded_text, seq_len)

                                        sub_span_prob = sub_head_prob * sub_h2t_prob + sub_tail_prob * sub_t2h_prob
                                        obj_span_prob = obj_head_prob * obj_h2t_prob + obj_tail_prob * obj_t2h_prob
                                        sub_trans_prob = sub_trans_head_prob * sub_trans_tail_prob
                                        obj_trans_prob = obj_trans_head_prob * obj_trans_tail_prob

                                        ep_score = self._cal_ep_score(sub_span_prob, obj_span_prob, sub_trans_prob, obj_trans_prob)
                                        if ep_score > 2.5:

                                            pred_rels_idx = self.cal_rel_prob(_sub_head_idx, _sub_tail_idx, _obj_head_idx, _obj_tail_idx, model, encoded_text)

                                            for rel_idx in pred_rels_idx:
                                                rel_idx = str(rel_idx)
                                                if (_sub_head_idx, _sub_tail_idx, _obj_head_idx, _obj_tail_idx, id2rel[rel_idx]) not in pred_eps:
                                                    pred_eps[(_sub_head_idx, _sub_tail_idx, _obj_head_idx, _obj_tail_idx, id2rel[rel_idx])] = ep_score
                                                else:
                                                    if ep_score > pred_eps[(_sub_head_idx, _sub_tail_idx, _obj_head_idx, _obj_tail_idx, id2rel[rel_idx])]:
                                                        pred_eps[(_sub_head_idx, _sub_tail_idx, _obj_head_idx, _obj_tail_idx, id2rel[rel_idx])] = ep_score

                else:
                    ent_bar = 0.5
                    rel_bar = 0.5
                    pred_eps = dict()
                    pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
                    sub_heads, sub_tails = np.where(pred_sub_heads.cpu()[0] > ent_bar)[0], \
                                           np.where(pred_sub_tails.cpu()[0] > ent_bar)[0]
                    subjects = []
                    for sub_head in sub_heads:
                        sub_tail = sub_tails[sub_tails >= sub_head]
                        if len(sub_tail) > 0:
                            sub_tail = sub_tail[0]
                            subject = tokens[sub_head: sub_tail]
                            subjects.append((subject, sub_head, sub_tail))
                    if subjects:
                        triple_list = []
                        # [subject_num, seq_len, bert_dim]
                        repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                        # [subject_num, 1, seq_len]
                        sub_head_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                        sub_tail_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                        for subject_idx, subject in enumerate(subjects):
                            sub_head_mapping[subject_idx][0][subject[1]] = 1
                            sub_tail_mapping[subject_idx][0][subject[2]] = 1
                        sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)
                        sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)
                        pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head_mapping,
                                                                                         sub_tail_mapping,
                                                                                         repeated_encoded_text)
                        for subject_idx, subject in enumerate(subjects):
                            obj_heads, obj_tails = np.where(pred_obj_heads.cpu()[subject_idx] > rel_bar), np.where(pred_obj_tails.cpu()[subject_idx] > rel_bar)
                            for obj_head, rel_head in zip(*obj_heads):
                                for obj_tail, rel_tail in zip(*obj_tails):
                                    if obj_head <= obj_tail and rel_head == rel_tail:
                                        ep_score = pred_obj_tails.cpu()[subject_idx][obj_head][rel_head].item()
                                        rel_head = str(rel_head)

                                        if (subject[1], subject[2], obj_head, obj_tail, id2rel[rel_head]) not in pred_eps:
                                            pred_eps[(subject[1], subject[2], obj_head, obj_tail, id2rel[rel_head])] = ep_score
                                        else:
                                            if ep_score > pred_eps[(subject[1], subject[2], obj_head, obj_tail, id2rel[rel_head])]:
                                                pred_eps[(subject[1], subject[2], obj_head, obj_tail, id2rel[rel_head])] = ep_score

                                        break

                for _ep in pred_eps:
                    pred_eps_id.append((_ep[0], _ep[1], _ep[2], _ep[3], pred_eps[_ep], data_id, _ep[4]))
                data_id += 1
                data = test_data_prefetcher.next()

        pred_eps_id = sorted(pred_eps_id, key=lambda x: x[4], reverse=True)


        def element_prf(pred_eps_id, data_gold, head_only=False, gold_type='EP', lang='ENGLISH'):

            correct_num, pred_num, gold_num = 0, 0, 0
            v_pred_entity_pair = set()

            #  To calculate gold number
            for item in data_gold:
                gold_triples = item[1]
                if gold_type == 'EP':
                    gold_info = _to_ep(gold_triples, head_only, lang=lang)
                elif gold_type == 'sub':
                    gold_info = _to_sub(gold_triples, head_only, lang=lang)
                elif gold_type == 'obj':
                    gold_info = _to_obj(gold_triples, head_only, lang=lang)
                elif gold_type == 'ALL':
                    gold_info = _to_triple(gold_triples, head_only, lang=lang)
                    # print(head_only, gold_info)
                gold_num += len(gold_info)

            # print("gold_type:{}, gold_num:{}".format(gold_type, gold_num))

            for _eps_id in pred_eps_id:
                gold_results, _tokens = _load_gold_data(data_gold, _eps_id[5], head_only, gold_type, lang=lang)

                sub = _tokens[_eps_id[0]: _eps_id[1]+1]
                sub = self.config.tokenizer.convert_tokens_to_string(sub)

                if lang is 'CHINESE':
                    sub = sub.replace(' ', '')
                    sub = sub.replace('$', ' ')

                sub = sub.strip().replace(" - ", "-")
                if head_only:
                    sub = sub.split(" ")[0]

                obj = _tokens[_eps_id[2]: _eps_id[3]+1]
                obj = self.config.tokenizer.convert_tokens_to_string(obj)
                # obj = ''.join([i.lstrip("##") for i in obj])
                #obj = ' '.join(obj.split('[unused1]'))
                obj = obj.strip().replace(" - ", "-")

                if lang is 'CHINESE':
                    obj = obj.replace(' ', '')
                    obj = obj.replace('$', ' ')

                if head_only:
                    obj = obj.split(" ")[0]

                rel = _eps_id[6]

                if gold_type == 'EP':
                    pred_info = (sub, obj, _eps_id[5])
                elif gold_type == 'sub':
                    pred_info = (sub, _eps_id[5])
                elif gold_type == 'obj':
                    pred_info = (obj, _eps_id[5])
                elif gold_type == 'ALL':
                    pred_info = (sub, rel, obj, _eps_id[5])

                if pred_info not in v_pred_entity_pair:
                    v_pred_entity_pair.add(pred_info)
                else:
                    continue

                if gold_type == 'EP':
                    pred_info = (sub, obj)
                elif gold_type == 'sub':
                    pred_info = (sub)
                elif gold_type == 'obj':
                    pred_info = (obj)
                elif gold_type == 'ALL':
                    pred_info = (sub, rel, obj)
                    # print(head_only, pred_info)

                if pred_info in gold_results:
                    correct_num += 1
                #else:
                #    if gold_type == 'ALL' and head_only == False:
                #         print("pred_info:{}".format(pred_info))
                #         print("gold_results:{}".format(gold_results))
                pred_num += 1

            p, r, f = _cal_prf(correct_num, pred_num, gold_num)
            print("gold_type:{} head_only:{} gold_num:{} pred_num:{} correct_num:{}, p:{},r:{},f:{},".format(gold_type, head_only, gold_num, pred_num, correct_num, p, r, f))

            return p, r, f

        # print(pred_eps_id)
        triple_p, triple_r, triple_f = element_prf(pred_eps_id, data_gold, gold_type='ALL', lang=self.config.dataset_lang)
        triple_hp, triple_hr, triple_hf = element_prf(pred_eps_id, data_gold, head_only=True, gold_type='ALL', lang=self.config.dataset_lang)

        ep_p, ep_r, ep_f = element_prf(pred_eps_id, data_gold, gold_type='EP', lang=self.config.dataset_lang)
        ep_hp, ep_hr, ep_hf = element_prf(pred_eps_id, data_gold, head_only=True, gold_type='EP', lang=self.config.dataset_lang)

        sub_p, sub_r, sub_f = element_prf(pred_eps_id, data_gold, gold_type='sub', lang=self.config.dataset_lang)
        sub_hp, sub_hr, sub_hf = element_prf(pred_eps_id, data_gold, head_only=True, gold_type='sub', lang=self.config.dataset_lang)

        obj_p, obj_r, obj_f = element_prf(pred_eps_id, data_gold, gold_type='obj', lang=self.config.dataset_lang)
        obj_hp, obj_hr, obj_hf = element_prf(pred_eps_id, data_gold, head_only=True, gold_type='obj', lang=self.config.dataset_lang)

        return triple_p, triple_r, triple_f, triple_hp, triple_hr, triple_hf, \
               ep_p, ep_r, ep_f, ep_hp, ep_hr, ep_hf, \
               sub_p, sub_r, sub_f, sub_hp, sub_hr, sub_hf, \
               obj_p, obj_r, obj_f, obj_hp, obj_hr, obj_hf

    def testall(self, model_pattern):

        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
        model.load_state_dict(torch.load(path))
        model.cuda()
        model.eval()

        test_data_loader = data_loader.get_loader(self.config, tokenizer=self.config.tokenizer, prefix=self.config.dev_prefix, is_test=True, collate_fn=partial(cmed_collate_fn, num_rels=self.config.rel_num))

        self.test(test_data_loader, self.config.step_dim, self.config.step_matrix, model)

        return
