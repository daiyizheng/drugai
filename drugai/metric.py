#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/22 22:12
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : metric.py
from __future__ import annotations, print_function
import argparse
import os,logging

import yaml
import pandas as pd
import torch

from drugai.metrics.metric_util import get_metrics
from drugai.utils.common import override_defaults, seed_everything

logger = logging.getLogger(__file__)


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default=None, type=str, required=True,
                        help="The parameter config path for samples.")
    return parser

"""
if ("FCD/Test" in config.metric_method or
        "FCD/TestSF" in config.metric_method or
        "SNN/Test" in config.metric_method or 
        "SNN/TestSF" in config.metric_method or 
        "Frag/Test" in config.metric_method or 
        "Frag/TestSF" in config.metric_method or 
        "Scaf/Test" in config.metric_method or 
        "Scaf/TestSF" in config.metric_method or 
        "novelty" in config.metric_method) \
            and config.metric_data_dir is None:
        raise KeyError
"""

def main():
    config = get_argparse().parse_args()
    config.no_cuda = False
    config.seed = 1314
    config.metric_save_dir = "./"
    config.model_name = "rr"
    config.gen_dir = "../results/RNNGenerate_results.csv"
    config.local_rank=-1
    config.metric_data_dir=None
    config.unique_k=[1000,10000]
    config.n_jobs=1
    config.batch_size=512
    # ## 修正的参数
    # f = open(config.config_dir, "r")
    # fix_args = yaml.full_load(f)
    # for k in fix_args:
    #     config = override_defaults(config, fix_args[k])

    ## 随机种子
    seed_everything(config.seed)

    # Setup CUDA， GPU & distributed training
    if config.local_rank == -1 or config.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        config.n_gpu = torch.cuda.device_count()
        logger.info({"n_gpu: ": config.n_gpu})
    else:  # initializes the distributed backend which will take care of suchronizing nodes/GPUs
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", config.local_rank)
        torch.distributed.init_porcess_group(backend="nccl")
        config.n_gpu = 1
    config.device = device

    metrics = get_metrics(config=config)
    table = pd.DataFrame([metrics]).T
    metrics_path = os.path.join(config.metric_save_dir, "metric_"+config.model_name+".csv")
    table.to_csv(metrics_path, header=False)

if __name__ == '__main__':
    main()

