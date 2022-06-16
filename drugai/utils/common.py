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
from drugai.vocab import Vocab


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


def default_collate_fn(vocabulary: Vocab):
    def collate_fn(data):
        data.sort(key=len, reverse=True)
        batch_token_ids = [vocabulary.string_to_ids(s) for s in data]
        batch_input_ids = [torch.tensor(b, dtype=torch.long)  for b in batch_token_ids]
        batch_source = pad_sequence([t[:-1] for t in batch_input_ids], batch_first=True, padding_value=vocabulary.pad_token_ids)
        batch_target = pad_sequence([t[1:] for t in batch_input_ids], batch_first=True, padding_value=vocabulary.pad_token_ids)
        batch_lengths = torch.tensor([len(t) - 1 for t in batch_input_ids], dtype=torch.long)
        return batch_source, batch_target, batch_lengths

    return collate_fn


def load_dataset(args, mode):
    processor = MODEL_CLASSES[args.model_name][1]()
    dataset = processor.get_dataset(args.data_dir, mode)
    return dataset
