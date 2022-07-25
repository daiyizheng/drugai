#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 11:23
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : loading.py
from __future__ import annotations, print_function

import os
from typing import Text, Any

import numpy as np
from moses.script_utils import read_smiles_csv


def training_data_from_paths(path: Text,
                             data_type: Text
                             ) -> Any:
    if path == None:
        return []
    if not os.path.exists(path):
        raise ValueError(f"File '{path}' does not exist.")
    return load_data(path=path, data_type=data_type)


def load_data(path: Text,
              data_type: Text
              ) -> np.ndarray:
    if data_type == "gen":
        return read_smiles_csv(path)
    else:
        raise KeyError
