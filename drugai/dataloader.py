#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 23:05
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : dataloader.py
from __future__ import annotations, print_function

import os
from typing import Text, Union, List

import pandas as pd
import numpy as np


class Processor(object):
    @staticmethod
    def read_csv(path: Text) -> np.ndarray:
        df = pd.read_csv(path)
        return df["SMILES"].values

    def get_dataset(self, data_dir, mode) -> Union[List, np.ndarray]:
        return self.read_csv(os.path.join(data_dir, mode + ".csv"))
