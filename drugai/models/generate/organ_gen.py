#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 15:58
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : organ.py
from __future__ import annotations, print_function
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class ORGAN(nn.Module):
    def __init__(self):
        super(ORGAN, self).__init__()
