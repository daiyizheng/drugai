#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 13:46
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : common.py
from __future__ import annotations, print_function

import copy
import os
import random
from typing import Optional, Any, Text, Dict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from drugai import MODEL_CLASSES


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def override_defaults(
        defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """
    Returns:
        updated config
    """
    if defaults:
        config = copy.deepcopy(defaults)
    else:
        config = {}

    if custom:
        for key in custom.keys():
            setattr(config, key, custom[key])

    return config