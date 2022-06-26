#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 0:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : io.py
from __future__ import annotations, print_function

from typing import Text
import os

import pandas as pd
import numpy as np

AVAILABLE_SPLITS = ['train', 'test', 'test_scaffolds']


def read_csv(path: Text):
    return pd.read_csv(path)

def get_dataset(split='train'):
    """
    code from moses:
    Loads MOSES dataset

    Arguments:
        split (str): split to load. Must be
            one of: 'train', 'test', 'test_scaffolds'

    Returns:
        list with SMILES strings
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}"
        )
    base_path = os.path.dirname(os.path.dirname(__file__))
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}")
    path = os.path.join(base_path, 'resources', split+'.csv.gz')
    smiles = pd.read_csv(path, compression='gzip')['SMILES'].values
    return smiles


def get_statistics(split='test'):
    """:arg
    code from moses:
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_path, 'resources', split+'_stats.npz')
    return np.load(path, allow_pickle=True)['stats'].item()
