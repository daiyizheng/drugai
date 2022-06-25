#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 23:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : dataset_base.py
from __future__ import annotations, print_function

import abc
import os
from functools import partial
from typing import Text, Union, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from drugai.models.vocab_base import Vocab
from drugai.utils.io import read_csv


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

class Processor(object):
    @staticmethod
    def read_csv(path: Text) -> np.ndarray:
        df = pd.read_csv(path)
        return df["SMILES"].values

    def get_dataset(self, data_dir, mode) -> Union[List, np.ndarray]:
        return self.read_csv(os.path.join(data_dir, mode + ".csv"))



class DataSetBase(abc.ABC):
    def __init__(self,
                 batch_size: int,
                 num_workers: int,
                 vocab: Vocab,
                 train_data: Text = None,
                 eval_data: Text = None,
                 test_data: Text = None,
                 collate_fn=None):

        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def prepare_data(self):
        # 下载
        pass

    def step(self, is_train=True, batch_size=None):
        if is_train:
            self.train_dataset = self.train_data
            self.eval_dataset = self.eval_data
        else:
            if batch_size is None:
                raise KeyError

            test_dataset = [torch.tensor([self.vocab.bos_token_ids], dtype=torch.long) for _ in range(batch_size)]
            test_dataset = torch.tensor(test_dataset, dtype=torch.long)
            self.test_dataset = test_dataset.unsqueeze(1)

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

# if __name__ == '__main__':
#
#     train_data = read_csv("../../datasets/train.csv")["SMILES"].values
#     vocab = Vocab.from_data(train_data)
#     fn = partial(default_collate_fn, vocab)
#
#     a = DataSetBase(train_data=train_data, batch_size=10, num_workers=1, collate_fn=fn)
#     a.step(is_train=True)
#     data = a.train_dataloader()
#     print(len(data))
#     for t in data:
#         print(t)
#         break
