#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 18:05
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : generate_dataset.py
from __future__ import annotations, print_function

import torch

from drugai.models.dataset import DataSet as ds


class GenDataset(ds):
    def __init__(self, bos_token_ids:int, **kwargs):
        super(GenDataset, self).__init__(**kwargs)
        self.bos_token_ids = bos_token_ids

    def step(self, is_train=True):
        if is_train:
            self.train_dataset = self.train_data
            self.eval_dataset = self.eval_data
        else:
            test_dataset = [torch.tensor([self.bos_token_ids], dtype=torch.long) for _ in range(self.batch_size)]
            test_dataset = torch.tensor(test_dataset, dtype=torch.long)
            self.test_dataset = test_dataset.unsqueeze(1)



