#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/24 22:03
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trains.py
from __future__ import annotations, print_function
import argparse
import logging
import os
from datetime import datetime

import torch
import yaml

from drugai.utils.common import override_defaults, seed_everything

logger = logging.getLogger(__file__)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default=None, type=str, required=True, help="The parameter config path.")

    return parser
def main():
    config = get_argparse().parse_args()
    ## 修正的参数
    f = open(config.config_dir, "r")
    fix_args = yaml.full_load(f)
    for k in fix_args:
        args = override_defaults(config, fix_args[k])

    ## 随机种子
    seed_everything(config.seed)

    ## 模型输出目录
    config.output_dir = os.path.join(config.output_dir,
                                   config.model_name.lower() + "_" + config.data_name + "_" + str(config.seed))
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    ## 日志输出
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    ## tensorboardx
    config.tensorboardx_path = os.path.join(config.tensorboardx_path, config.model_name.lower() + \
                                          "_" + config.data_name + "_" + str(config.seed) + \
                                          "_" + datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(config.tensorboardx_path):
        os.makedirs(config.tensorboardx_path)

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






if __name__ == '__main__':
    pass