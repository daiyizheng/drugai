#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 17:59
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : dataset.py
from __future__ import annotations, print_function

import torch
from torch.nn.utils.rnn import pad_sequence


def default_collate_fn(vocab, data):
    data.sort(key=len, reverse=True)
    batch_token_ids = [vocab.string_to_ids(s) for s in data]
    batch_input_ids = [torch.tensor(b, dtype=torch.long) for b in batch_token_ids]
    batch_source = pad_sequence([t[:-1] for t in batch_input_ids], batch_first=True,
                                padding_value=vocab.pad_token_ids)
    batch_target = pad_sequence([t[1:] for t in batch_input_ids], batch_first=True,
                                padding_value=vocab.pad_token_ids)
    batch_lengths = torch.tensor([len(t) - 1 for t in batch_input_ids], dtype=torch.long)
    return batch_source, batch_target, batch_lengths


def single_collate_fn(vocab, data):
    data.sort(key=len, reverse=True)
    batch_token_ids = [vocab.string_to_ids(s) for s in data]
    batch_input_ids = [torch.tensor(b, dtype=torch.long) for b in batch_token_ids]
    return batch_input_ids
