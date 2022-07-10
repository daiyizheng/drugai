#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 17:59
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : dataset.py
from __future__ import annotations, print_function

import abc
from typing import Text

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def default_collate_fn(vocab, data):
    data.sort(key=len, reverse=True)
    batch_token_ids = [vocab.string_to_ids(s) for s in data]
    batch_input_ids = [torch.tensor(b, dtype=torch.long) for b in batch_token_ids]
    batch_source = pad_sequence([t[:-1] for t in batch_input_ids], batch_first=True,
                                padding_value=vocab.pad_token_ids)
    batch_target = pad_sequence([t[1:] for t in batch_input_ids], batch_first=True,
                                padding_value=vocab.pad_token_ids)
    batch_lengths = torch.tensor([len(t) - 1 for t in batch_input_ids], dtype=torch.long)
    return batch_source, batch_target, batch_lengths


class DataSet(abc.ABC):
    def __init__(self,
                 batch_size: int,
                 num_workers: int,
                 train_data: Text = None,
                 eval_data: Text = None,
                 test_data: Text = None,
                 collate_fn=None):
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def prepare_data(self):
        # 下载
        raise NotImplemented

    def step(self, is_train=True):
        raise NotImplemented

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def eval_dataloader(self):
        return DataLoader(self.eval_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=None)
