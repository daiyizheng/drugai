#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 21:25
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trainer.py
from __future__ import annotations, print_function

from drugai.models.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self,
                 model,
                 vocab,
                 config,
                 collate_fn=None,
                 train_dataset=None,
                 eval_dataset=None,
                 test_dataset=None
                 ):
        super(Trainer, self).__init__(model=model,
                                      vocab=vocab,
                                      config=config,
                                      collate_fn=collate_fn,
                                      train_dataset=train_dataset,
                                      eval_dataset=eval_dataset,
                                      test_dataset=test_dataset)

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

