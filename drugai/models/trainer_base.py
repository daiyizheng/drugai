#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/22 11:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trainer_base.py
from __future__ import annotations, print_function

import abc
import os, logging

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from drugai.models.dataset_base import DataSetBase


logger = logging.getLogger(__file__)


class TrainerBase(abc.ABC):
    def __init__(self,config):
        self.config = config
        self.logs = {}
        self.compute_metric = None
        self.global_step = 1

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs)
        return logits

    @abc.abstractmethod
    def config_optimizer(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def config_criterion(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        self.tb_writer = SummaryWriter(self.config.tensorboardx_path)
        train_dataloader = self.get_train_dataloader()
        t_total = len(train_dataloader) // self.config.gradient_accumulation_steps * self.config.epochs
        self.optimizer, self.scheduler = self.config_optimizer()
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        logger.info("  Num Epochs = %d", self.config.epochs)
        logger.info("  Instantaneous batch size GPU = %d", self.config.batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.config.batch_size * self.config.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if self.config.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.model.zero_grad()

        for epoch in range(self.config.epochs):
            self.logs = {"loss": 0.0, "eval_loss":0.0}
            self.epoch_data = tqdm(train_dataloader, desc='Training (epoch #{})'.format(epoch))
            self.model.train()
            self.train_epoch()
            self.evaluate()
            for key, value in self.logs.items():
                self.tb_writer.add_scalar(key, value, epoch)

    @abc.abstractmethod
    def train_epoch(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, *args, **kwargs):
        eval_dataloader = self.get_evaluate_dataloader()
        # Eval!
        logger.info("***** Running evaluation on dev dataset *****")
        logger.info("  Num examples = %d", len(eval_dataloader))
        logger.info("  Batch size = %d", self.config.batch_size)
        self.eval_data = tqdm(eval_dataloader, desc='Evaluation')
        self.evaluate_epoch()

    @abc.abstractmethod
    def evaluate_step(self,*args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_epoch(self,*args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        raise  NotImplementedError

    def save_checkpoint(self, model, vocab, config):

        model_to_save = model.module if hasattr(model, 'module') else model
        ## 保存模型
        torch.save(model_to_save.state_dict(), os.path.join(self.config.output_dir, "model.pt"))
        ## 保存字典
        torch.save(vocab, os.path.join(self.config.output_dir, "vocab.pt"))
        ## 保存参数
        torch.save(config, os.path.join(self.config.output_dir, "args.pt"))

    def get_train_dataloader(self, *args, **kwargs):
        raise NotImplementedError

    def get_evaluate_dataloader(self, *args, **kwargs):
        raise NotImplementedError

    def get_predict_dataloader(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self,*args, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            raise KeyError
        dataset = kwargs.get("dataset", None)
        if dataset is None:
            raise KeyError
        criterion = kwargs.get("criterion", None) or self.config_criterion(*args, **kwargs)
        if criterion is None:
            raise KeyError
        vocab = kwargs.get("vocab", None)
        if vocab is None:
            raise  KeyError

        self.criterion = criterion
        self.model = model
        self.dataset:DataSetBase = dataset
        self.compute_metric = kwargs.get("compute_metric", None)
        self.vocab = vocab
        self.dataset.step(is_train=True)
        self.train(*args, **kwargs)

    def sample(self, n_sample,*args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
