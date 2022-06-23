#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/22 11:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trainer_base.py
from __future__ import annotations, print_function

import abc
import os

import torch


class TrainerBase(abc.ABC):
    def __init__(self,
                 model,
                 vocab,
                 config,
                 collate_fn=None,
                 train_dataset=None,
                 eval_dataset=None,
                 test_dataset=None
                 ):
        self.model = model
        self.vocab = vocab
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        self.model.to(config.device)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def config_optimizer(self, *args, **kwargs):
        raise  NotImplementedError

    @torch.no_grad()
    def evaluate(self):
        raise  NotImplementedError

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        raise  NotImplementedError

    @classmethod
    def load_model(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def save_model(self):

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        ## 保存模型
        torch.save(model_to_save.state_dict(), os.path.join(self.config.output_dir, "model.pt"))
        ## 保存字典
        torch.save(self.vocab, os.path.join(self.config.output_dir, "vocab.pt"))
        ## 保存参数
        torch.save(self.config, os.path.join(self.config.output_dir, "args.pt"))

    def get_train_dataloader(self, *args, **kwargs):
        raise  NotImplementedError

    def get_evaluate_dataloader(self, *args, **kwargs):
        raise  NotImplementedError

    def get_predict_dataloader(self, *args, **kwargs):
        raise  NotImplementedError