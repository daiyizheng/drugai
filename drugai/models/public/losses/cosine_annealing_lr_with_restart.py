#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 23:49
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : cosine_annealing_lr_with_restart.py
from __future__ import annotations, print_function

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self,
                 optimizer:Optimizer,
                 lr_n_period:int,
                 lr_n_mult:int,
                 lr_end:float) -> None:
        self.n_period = lr_n_period #
        self.n_mult = lr_n_mult #
        self.lr_end = lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self) -> List:#
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self,
             epoch=None
             )->None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end