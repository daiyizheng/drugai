#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 0:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : io.py
from __future__ import annotations, print_function

from typing import Text
import pandas as pd


def read_csv(path: Text):
    return pd.read_csv(path)
