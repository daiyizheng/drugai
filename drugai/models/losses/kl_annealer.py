#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 23:48
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : kl_annealer.py
from __future__ import annotations, print_function


class KLAnnealer:
    def __init__(self,
                 epochs: int,
                 kl_start: int,
                 kl_w_start: float,
                 kl_w_end: float
                 ) -> None:
        self.i_start = kl_start  # epoch起点
        self.w_start = kl_w_start  # 权重起始值
        self.w_max = kl_w_end  # 最大权重值
        self.epochs = epochs

        self.inc = (self.w_max - self.w_start) / (self.epochs - self.i_start)

    def __call__(self,
                 i: int
                 ) -> float:
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc
