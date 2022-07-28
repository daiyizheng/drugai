#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 13:21
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : training_data.py
from __future__ import annotations, print_function

import logging
from typing import Text, Any

from torch.utils.data import DataLoader

from drugai.models.vocab import Vocab

logger = logging.getLogger(__name__)


class TrainingData:
    def __init__(self,
                 train_data,
                 eval_data = None,
                 test_data = None,
                 num_workers: int = 0,
                 **kwargs):
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.num_workers = num_workers

    def dataloader(self,
                   batch_size: int,
                   collate_fn: Any,
                   shuffle: bool = True,
                   mode: Text = "train"):
        if mode == "train":
            dataset = self.train_data
        elif mode == "eval":
            dataset = self.eval_data
        elif mode == "test":
            dataset = self.test_data
        else:
            raise KeyError

        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn)

    def build_vocab(self,
                    vocab:Vocab
                    ) -> "Vocab":
        return vocab.from_data(self.train_data)
