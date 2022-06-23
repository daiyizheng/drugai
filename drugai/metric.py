#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/22 22:12
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : metric.py
from __future__ import annotations, print_function
import argparse
import pandas as pd
from moses.metrics import get_all_metrics

gen = pd.read_csv("../experiments/outputs/smiles.csv")

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_config_dir", default=None, type=str, required=True,
                        help="The parameter config path for samples.")
    return parser


metrics = get_all_metrics(gen=gen, k=[1000, 10000], n_jobs=1,
                              device="cpu",
                              test_scaffolds=None,
                              ptest=None, ptest_scaffolds=None,
                              test=None, train=None)