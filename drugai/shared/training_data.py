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

logger = logging.getLogger(__name__)


class TrainingData:
    def __init__(self,
                 train_data,
                 eval_data=None,
                 test_data=None):
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

    def dataloader(self,
                   batch_size: int,
                   num_workers: int,
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
                          num_workers=num_workers,
                          collate_fn=collate_fn)