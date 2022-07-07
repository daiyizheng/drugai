#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 23:08
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : vae.py
from __future__ import annotations, print_function

import torch.nn as nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
