from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
import numpy as np
from random import choice

BERT_MAX_LEN = 512


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class JEDataset(Dataset):
    def __init__(self, config, prefix, is_test, tokenizer):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.step_dim = config.step_dim

        print(prefix)
        if self.config.debug:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))[:500]
        else:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[1]

        print(len(self.json_data))

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        ins_json_data = self.json_data[idx]
        text = ins_json_data['text']
        if self.config.dataset_lang is 'CHINESE':
            text = text.replace(' ', '$')

        text = ' '.join(text.split()[:self.config.max_len])
        tokens = self.tokenizer.tokenize(text, add_special_tokens=True)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[: BERT_MAX_LEN]
        text_len = len(tokens)
        rel_num = len(self.rel2id)

        # sent_label
        sent_label = np.zeros(rel_num)

        if not self.is_test:

            s2ro_map = {}
            o2rs_map = {}

            h2t_sub_map = {}
            t2h_sub_map = {}

            h2t_obj_map = {}
            t2h_obj_map = {}

            ep2r_map = {}

            flag = False
            for triple in ins_json_data['triple_list']:
                if self.config.dataset_lang is 'CHINESE':
                    triple = (triple[0].replace(' ', '$'), triple[1], triple[2].replace(' ', '$'))

                if flag is False:
                    if triple[2] == 'object_placeholder' or triple[0] == 'subject_placeholder':
                        flag = True

                triple = (self.tokenizer.tokenize(triple[0]), triple[1], self.tokenizer.tokenize(triple[2]))
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])

                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub_tail_idx = sub_head_idx + len(triple[0]) - 1
                    obj_tail_idx = obj_head_idx + len(triple[2]) - 1
                    sub = (sub_head_idx, sub_tail_idx)
                    obj = (obj_head_idx, obj_tail_idx)
                    entity_pair = (sub_head_idx, sub_tail_idx, obj_head_idx, obj_tail_idx)
                    # print("triple:{}".format(triple))
                    # print("entity_pair:{}".format(entity_pair))

                    # entity pair
                    if entity_pair not in ep2r_map:
                        ep2r_map[entity_pair] = []
                    ep2r_map[entity_pair].append(self.rel2id[triple[1]])

                    # sub head to tail
                    if sub_head_idx not in h2t_sub_map:
                        h2t_sub_map[sub_head_idx] = []
                    h2t_sub_map[sub_head_idx].append(sub_tail_idx)

                    # sub tail to head
                    if sub_tail_idx not in t2h_sub_map:
                        t2h_sub_map[sub_tail_idx] = []
                    t2h_sub_map[sub_tail_idx].append(sub_head_idx)

                    # obj head to tail
                    if obj_head_idx not in h2t_obj_map:
                        h2t_obj_map[obj_head_idx] = []
                    h2t_obj_map[obj_head_idx].append(obj_tail_idx)

                    # obj tail to head
                    if obj_tail_idx not in t2h_obj_map:
                        t2h_obj_map[obj_tail_idx] = []
                    t2h_obj_map[obj_tail_idx].append(obj_head_idx)

                    # sub to obj
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, obj_tail_idx, self.rel2id[triple[1]]))

                    # obj to sub
                    if obj not in o2rs_map:
                        o2rs_map[obj] = []
                    o2rs_map[obj].append((sub_head_idx, sub_tail_idx, self.rel2id[triple[1]]))

            if s2ro_map:

                # sub_head_walk_step, sub_tail_walk_step, obj_head_walk_step, obj_tail_walk_step = np.zeros(text_len), np.zeros(text_len), np.zeros(text_len), np.zeros(text_len)

                tokenizer_dict = self.tokenizer(text)
                token_ids = np.array(tokenizer_dict['input_ids'])
                masks = np.array(tokenizer_dict['attention_mask'])
                # print("token_ids:{}".format(token_ids))
                # print("masks:{}".format(masks))


                # used for predicting all subjects and objects.
                em_sub_heads, em_sub_tails = np.zeros(text_len), np.zeros(text_len)
                for _s in s2ro_map:
                    em_sub_heads[_s[0]] = 1
                    em_sub_tails[_s[1]] = 1

                em_obj_heads, em_obj_tails = np.zeros(text_len), np.zeros(text_len)
                for _o in o2rs_map:
                    em_obj_heads[_o[0]] = 1
                    em_obj_tails[_o[1]] = 1

                # sample subject
                sample_sub_head_idx, sample_sub_tail_idx = choice(list(s2ro_map.keys()))
                sample_sub_head, sample_sub_tail = np.zeros(text_len), np.zeros(text_len)
                sample_sub_head[sample_sub_head_idx] = 1
                sample_sub_tail[sample_sub_tail_idx] = 1
                sample_sub_len = sample_sub_tail_idx - sample_sub_head_idx + 1

                # subject to (relation-object)
                s2ro_heads, s2ro_tails = np.zeros((text_len, self.config.rel_num)), np.zeros((text_len, self.config.rel_num))
                for ro in s2ro_map.get((sample_sub_head_idx, sample_sub_tail_idx), []):
                    s2ro_heads[ro[0]][ro[2]] = 1
                    s2ro_tails[ro[1]][ro[2]] = 1

                # used for predicting span of sampled subject.
                sub_h2t, sub_t2h = np.zeros(text_len), np.zeros(text_len)
                for _t_idx in h2t_sub_map[sample_sub_head_idx]:
                    sub_h2t[_t_idx] = 1
                for _h_idx in t2h_sub_map[sample_sub_tail_idx]:
                    sub_t2h[_h_idx] = 1

                # sample object
                sample_ro = choice(s2ro_map.get((sample_sub_head_idx, sample_sub_tail_idx), []))
                sample_obj_head_idx = sample_ro[0]
                sample_obj_tail_idx = sample_ro[1]
                sample_obj_len = sample_obj_tail_idx - sample_obj_head_idx + 1

                sample_obj_head, sample_obj_tail = np.zeros(text_len), np.zeros(text_len)
                sample_obj_head[sample_obj_head_idx] = 1
                sample_obj_tail[sample_obj_tail_idx] = 1

                # make step.
                def make_step(sample_idx, text_len, step_dim, step_matrix):
                    walk_step = np.zeros((text_len, step_dim))
                    for i in range(text_len):
                        # print(i, sample_idx, self.max_len)
                        walk_step[i] = step_matrix[i - sample_idx + self.max_len]
                    return walk_step

                sub_head_walk_step = make_step(sample_sub_head_idx, text_len, self.config.step_dim,
                                               self.config.step_matrix)
                sub_tail_walk_step = make_step(sample_sub_tail_idx, text_len, self.config.step_dim,
                                               self.config.step_matrix)
                obj_head_walk_step = make_step(sample_obj_head_idx, text_len, self.config.step_dim,
                                               self.config.step_matrix)
                obj_tail_walk_step = make_step(sample_obj_tail_idx, text_len, self.config.step_dim,
                                               self.config.step_matrix)

                # used for predicting span of sampled object.
                obj_h2t, obj_t2h = np.zeros(text_len), np.zeros(text_len)
                for _t_idx in h2t_obj_map[sample_obj_head_idx]:
                    obj_h2t[_t_idx] = 1
                for _h_idx in t2h_obj_map[sample_obj_tail_idx]:
                    obj_t2h[_h_idx] = 1

                # used for predicting object and subject.
                sub2obj_h, sub2obj_t, obj2sub_h, obj2sub_t = np.zeros(text_len), np.zeros(text_len), np.zeros(
                    text_len), np.zeros(text_len)

                for _obj in s2ro_map[(sample_sub_head_idx, sample_sub_tail_idx)]:
                    sub2obj_h[_obj[0]] = sub2obj_t[_obj[1]] = 1

                for _sub in o2rs_map[(sample_obj_head_idx, sample_obj_tail_idx)]:
                    obj2sub_h[_sub[0]] = obj2sub_t[_sub[1]] = 1

                # relation info
                rm_relation_labels = np.zeros(self.config.rel_num)
                for _r in ep2r_map.get(
                        (sample_sub_head_idx, sample_sub_tail_idx, sample_obj_head_idx, sample_obj_tail_idx), []):
                    rm_relation_labels[_r] = 1

                return token_ids, masks, text_len, sent_label, \
                       sample_sub_head_idx, sample_sub_tail_idx, sample_obj_head_idx, sample_obj_tail_idx, \
                       s2ro_heads, s2ro_tails, \
                       em_sub_heads, em_sub_tails, em_obj_heads, em_obj_tails, \
                       sample_sub_head, sample_sub_tail, sample_obj_head, sample_obj_tail, \
                       sub_h2t, sub_t2h, obj_h2t, obj_t2h, \
                       sub2obj_h, sub2obj_t, obj2sub_h, obj2sub_t, sample_sub_len, sample_obj_len, \
                       sub_head_walk_step, sub_tail_walk_step, obj_head_walk_step, obj_tail_walk_step, rm_relation_labels, \
                       ins_json_data['triple_list'], ins_json_data['text'], tokens

            else:
                return None
        else:
            # print(tokens)
            tokenizer_dict = self.tokenizer(text)
            token_ids = np.array(tokenizer_dict['input_ids'])
            masks = np.array(tokenizer_dict['attention_mask'])

            sample_sub_head, sample_sub_tail = np.zeros(text_len), np.zeros(text_len)
            sample_obj_head, sample_obj_tail = np.zeros(text_len), np.zeros(text_len)
            em_sub_heads, em_sub_tails = np.zeros(text_len), np.zeros(text_len)
            em_obj_heads, em_obj_tails = np.zeros(text_len), np.zeros(text_len)

            s2ro_heads, s2ro_tails = np.zeros((text_len, self.config.rel_num)), np.zeros(
                (text_len, self.config.rel_num))

            sample_sub_head_idx = 0
            sample_sub_tail_idx = 0
            sample_obj_head_idx = 0
            sample_obj_tail_idx = 0
            sample_sub_len = 0
            sample_obj_len = 0
            sub_h2t, sub_t2h = np.zeros(text_len), np.zeros(text_len)
            obj_h2t, obj_t2h = np.zeros(text_len), np.zeros(text_len)
            sub2obj_h, sub2obj_t, obj2sub_h, obj2sub_t = np.zeros(text_len), np.zeros(text_len), np.zeros(
                text_len), np.zeros(text_len)
            rm_relation_labels = np.zeros(self.config.rel_num)
            sub_head_walk_step, sub_tail_walk_step, obj_head_walk_step, obj_tail_walk_step = np.zeros(
                (text_len, self.step_dim)), np.zeros((text_len, self.step_dim)), np.zeros(
                (text_len, self.step_dim)), np.zeros((text_len, self.step_dim))

            return token_ids, masks, text_len, sent_label, \
                   sample_sub_head_idx, sample_sub_tail_idx, sample_obj_head_idx, sample_obj_tail_idx, \
                   s2ro_heads, s2ro_tails, \
                   em_sub_heads, em_sub_tails, em_obj_heads, em_obj_tails, \
                   sample_sub_head, sample_sub_tail, sample_obj_head, sample_obj_tail, \
                   sub_h2t, sub_t2h, obj_h2t, obj_t2h, \
                   sub2obj_h, sub2obj_t, obj2sub_h, obj2sub_t, sample_sub_len, sample_obj_len, \
                   sub_head_walk_step, sub_tail_walk_step, obj_head_walk_step, obj_tail_walk_step, rm_relation_labels, \
                   ins_json_data['triple_list'], ins_json_data['text'], tokens


def cmed_collate_fn(batch, num_rels=21):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)

    token_ids, masks, text_len, sent_label, \
    sample_sub_head_idx, sample_sub_tail_idx, sample_obj_head_idx, sample_obj_tail_idx, \
    s2ro_heads, s2ro_tails, \
    em_sub_heads, em_sub_tails, em_obj_heads, em_obj_tails, \
    sample_sub_head, sample_sub_tail, sample_obj_head, sample_obj_tail, \
    sub_h2t, sub_t2h, obj_h2t, obj_t2h, \
    sub2obj_h, sub2obj_t, obj2sub_h, obj2sub_t, sample_sub_len, sample_obj_len, \
    sub_head_walk_step, sub_tail_walk_step, obj_head_walk_step, obj_tail_walk_step, rm_relation_labels, \
    triples, text, tokens = zip(*batch)

    # print(sent_rel_labels)
    cur_batch = len(batch)
    max_text_len = max(text_len)
    step_len = len(sub_head_walk_step[0][0])

    batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()

    batch_rel_labels = torch.Tensor(cur_batch, num_rels).zero_()

    batch_sent_label = torch.Tensor(cur_batch, num_rels).zero_()

    batch_em_sub_heads = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_em_sub_tails = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_em_obj_heads = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_em_obj_tails = torch.Tensor(cur_batch, max_text_len).zero_()

    batch_sample_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sample_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sample_obj_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sample_obj_tail = torch.Tensor(cur_batch, max_text_len).zero_()

    batch_sub_h2t = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_t2h = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_obj_h2t = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_obj_t2h = torch.Tensor(cur_batch, max_text_len).zero_()

    batch_s2ro_heads = torch.Tensor(cur_batch, max_text_len, num_rels).zero_()
    batch_s2ro_tails = torch.Tensor(cur_batch, max_text_len, num_rels).zero_()

    batch_sub2obj_h = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub2obj_t = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_obj2sub_h = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_obj2sub_t = torch.Tensor(cur_batch, max_text_len).zero_()

    batch_sub_head_idx = torch.LongTensor(cur_batch).zero_()
    batch_sub_tail_idx = torch.LongTensor(cur_batch).zero_()
    batch_obj_head_idx = torch.LongTensor(cur_batch).zero_()
    batch_obj_tail_idx = torch.LongTensor(cur_batch).zero_()

    batch_sample_sub_len = [0] * cur_batch
    batch_sample_obj_len = [0] * cur_batch

    batch_sub_head_walk_step = torch.Tensor(cur_batch, max_text_len, step_len).zero_()
    batch_sub_tail_walk_step = torch.Tensor(cur_batch, max_text_len, step_len).zero_()
    batch_obj_head_walk_step = torch.Tensor(cur_batch, max_text_len, step_len).zero_()
    batch_obj_tail_walk_step = torch.Tensor(cur_batch, max_text_len, step_len).zero_()

    for i in range(cur_batch):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))

        batch_sent_label[i, :].copy_(torch.from_numpy(sent_label[i]))

        batch_em_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(em_sub_heads[i]))
        batch_em_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(em_sub_tails[i]))
        batch_em_obj_heads[i, :text_len[i]].copy_(torch.from_numpy(em_obj_heads[i]))
        batch_em_obj_tails[i, :text_len[i]].copy_(torch.from_numpy(em_obj_tails[i]))

        batch_sample_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sample_sub_head[i]))
        batch_sample_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sample_sub_tail[i]))
        batch_sample_obj_head[i, :text_len[i]].copy_(torch.from_numpy(sample_obj_head[i]))
        batch_sample_obj_tail[i, :text_len[i]].copy_(torch.from_numpy(sample_obj_tail[i]))

        batch_s2ro_heads[i, :text_len[i], :].copy_(torch.from_numpy(s2ro_heads[i]))
        batch_s2ro_tails[i, :text_len[i], :].copy_(torch.from_numpy(s2ro_tails[i]))

        batch_sub_h2t[i, :text_len[i]].copy_(torch.from_numpy(sub_h2t[i]))
        batch_sub_t2h[i, :text_len[i]].copy_(torch.from_numpy(sub_t2h[i]))
        batch_obj_h2t[i, :text_len[i]].copy_(torch.from_numpy(obj_h2t[i]))
        batch_obj_t2h[i, :text_len[i]].copy_(torch.from_numpy(obj_t2h[i]))

        batch_sub2obj_h[i, :text_len[i]].copy_(torch.from_numpy(sub2obj_h[i]))
        batch_sub2obj_t[i, :text_len[i]].copy_(torch.from_numpy(sub2obj_t[i]))
        batch_obj2sub_h[i, :text_len[i]].copy_(torch.from_numpy(obj2sub_h[i]))
        batch_obj2sub_t[i, :text_len[i]].copy_(torch.from_numpy(obj2sub_t[i]))

        batch_sub_head_idx[i] = sample_sub_head_idx[i]
        batch_sub_tail_idx[i] = sample_sub_tail_idx[i]
        batch_obj_head_idx[i] = sample_obj_head_idx[i]
        batch_obj_tail_idx[i] = sample_obj_tail_idx[i]

        batch_sample_sub_len[i] = sample_sub_len
        batch_sample_obj_len[i] = sample_obj_len

        batch_sub_head_walk_step[i, :text_len[i], :].copy_(torch.from_numpy(sub_head_walk_step[i]))
        batch_sub_tail_walk_step[i, :text_len[i], :].copy_(torch.from_numpy(sub_tail_walk_step[i]))
        batch_obj_head_walk_step[i, :text_len[i], :].copy_(torch.from_numpy(obj_head_walk_step[i]))
        batch_obj_tail_walk_step[i, :text_len[i], :].copy_(torch.from_numpy(obj_tail_walk_step[i]))

        batch_rel_labels[i, :].copy_(torch.from_numpy(rm_relation_labels[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'sent_rel': batch_sent_label,
            'em_sub_heads': batch_em_sub_heads,
            'em_sub_tails': batch_em_sub_tails,
            'em_obj_heads': batch_em_obj_heads,
            'em_obj_tails': batch_em_obj_tails,
            'sample_sub_head': batch_sample_sub_head,
            'sample_sub_tail': batch_sample_sub_tail,
            'sample_obj_head': batch_sample_obj_head,
            'sample_obj_tail': batch_sample_obj_tail,
            'batch_s2ro_heads': batch_s2ro_heads,
            'batch_s2ro_tails': batch_s2ro_tails,
            'sample_sub_head_idx': batch_sub_head_idx,
            'sample_sub_tail_idx': batch_sub_tail_idx,
            'sample_obj_head_idx': batch_obj_head_idx,
            'sample_obj_tail_idx': batch_obj_tail_idx,
            'sub_head_walk_step': batch_sub_head_walk_step,
            'sub_tail_walk_step': batch_sub_tail_walk_step,
            'obj_head_walk_step': batch_obj_head_walk_step,
            'obj_tail_walk_step': batch_obj_tail_walk_step,
            'sample_obj_len': batch_sample_obj_len,
            'sample_sub_len': batch_sample_sub_len,
            'sub_h2t': batch_sub_h2t,
            'sub_t2h': batch_sub_t2h,
            'obj_h2t': batch_obj_h2t,
            'obj_t2h': batch_obj_t2h,
            'sub2obj_h': batch_sub2obj_h,
            'sub2obj_t': batch_sub2obj_t,
            'obj2sub_h': batch_obj2sub_h,
            'obj2sub_t': batch_obj2sub_t,
            'rel_labels': batch_rel_labels,
            'triples': triples,
            'tokens': tokens,
            'text': text}


def get_loader(config, prefix, tokenizer=None, is_test=False, num_workers=0, collate_fn=cmed_collate_fn):
    dataset = JEDataset(config, prefix, is_test, tokenizer)
    if not is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)

    return data_loader


# get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=partial(cmed_collate_fn, num_rels=12))


class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data