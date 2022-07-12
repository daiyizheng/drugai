#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 13:42
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : generate_component.py
from __future__ import annotations, print_function

from typing import Optional, Dict, Text, Any
import logging

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from drugai.component import Component

logger = logging.getLogger(__name__)


class GenerateComponent(Component):
    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 tensorboardx_dir: Text = None,
                 no_cuda: bool = True,
                 local_rank: int = -1,
                 fp16: bool = False,
                 fp16_opt_level: Text = "01",  # '00', '01', '02', '03'
                 **kwargs:Any):
        super(GenerateComponent, self).__init__(component_config, **kwargs)
        ##  device
        if local_rank == -1 or no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            torch.distributed.init_porcess_group(backend="nccl")
            self.n_gpu = 1
        logger.info({"n_gpu: ": self.n_gpu})
        self.tb_writer = None
        if tensorboardx_dir is not None:
            self.tb_writer = SummaryWriter(tensorboardx_dir)
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level

    def config_optimizer(self, *args, **kwargs):
        raise NotImplemented

    def config_criterion(self, *args, **kwargs):
        raise NotImplemented

    def train(self, *args:Any, **kwargs:Any):
        raise NotImplemented

    def train_epoch(self, *args, **kwargs):
        raise NotImplemented

    def train_step(self, *args, **kwargs):
        raise NotImplemented

    @torch.no_grad()
    def evaluate(self, *args, **kwargs):
        raise NotImplemented

    @torch.no_grad()
    def evaluate_step(self, *args, **kwargs):
        raise NotImplemented

    @torch.no_grad()
    def evaluate_epoch(self, *args, **kwargs):
        raise NotImplemented

    @torch.no_grad()
    def predict_epoch(self, *args, **kwargs):
        raise  NotImplemented

    def predict(self, *args, **kwargs):
        raise  NotImplemented

    def get_train_dataloader(self, *args, **kwargs):
        raise NotImplemented

    def get_evaluate_dataloader(self, *args, **kwargs):
        raise NotImplemented

    def get_predict_dataloader(self, *args, **kwargs):
        raise NotImplemented

    def sample(self, n_sample, *args, **kwargs):
        raise NotImplemented

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs)
        return logits

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
