#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 17:59
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : dataset.py
from __future__ import annotations, print_function
import logging
from typing import Dict, Text, Any, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from drugai.shared.message import Message

from drugai.shared.preprocess.utils import one_hot

logger = logging.getLogger(__name__)


def default_collate_fn(vocab, messages:List[Message]):
    data = [m.smiles for m in messages]
    data.sort(key=len, reverse=True)
    batch_token_ids = [vocab.string_to_ids(s) for s in data]
    batch_input_ids = [torch.tensor(b, dtype=torch.long) for b in batch_token_ids]
    batch_source = pad_sequence([t[:-1] for t in batch_input_ids], batch_first=True,
                                padding_value=vocab.pad_token_ids)
    batch_target = pad_sequence([t[1:] for t in batch_input_ids], batch_first=True,
                                padding_value=vocab.pad_token_ids)
    batch_lengths = torch.tensor([len(t) - 1 for t in batch_input_ids], dtype=torch.long)
    return batch_source, batch_target, batch_lengths


def single_collate_fn(vocab, messages:List[Message]):
    data = [m.smiles for m in messages]
    data.sort(key=len, reverse=True)
    batch_token_ids = [vocab.string_to_ids(s) for s in data]
    batch_input_ids = [torch.tensor(b, dtype=torch.long) for b in batch_token_ids]
    return batch_input_ids


def moflow_collate_fn(num_max_id:List,
                      out_size:int,
                      data:torch.tensor):
    node_array = []
    adj_array = []
    for d in data:# [batch]
        node, adj = d   # node (9,), adj (4,9,9), label (15,)
        # convert to one-hot vector
        node = one_hot(num_max_id=num_max_id, out_size=out_size, data=node).astype(np.float32)
        # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond. 单，双，三和无键。 请注意，最后一个通道轴没有连接而不是芳香键。
        adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0).astype(np.float32)
        node_array.append(node)
        adj_array.append(adj)
    node_array = np.array(node_array)
    adj_array = np.array(adj_array)
    return node_array, adj_array

class NumpyTupleDataset(Dataset):
    def __init__(self, 
                 datasets:Union[List[Message], List[np.ndarray]],
                 **kwargs):
        if isinstance(datasets[0], Message):
            atom_matrix = []
            adj_matrix = []
            for d in datasets:
                atom_matrix.append(d.atom_matrix)
                adj_matrix.append(d.adj_matrix)   
        datasets = [atom_matrix, adj_matrix]
        length = len(datasets[0])  # 133885
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        # Initialization
        self._datasets = datasets
        self._length = length
    
    def __len__(self
                )->int:
        return self._length
    
    def __getitem__(self, index:int):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, np.ndarray)):
            length = len(batches[0])
            batches = [tuple([batch[i] for batch in batches])
                    for i in range(length)]   # six.moves.range(length)]
        else:
            batches = tuple(batches)
        return batches
    
    @property
    def get_datasets(self):
        return self._datasets