#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 0:52
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : drugai.py
from __future__ import annotations, print_function

import logging
import os
from datetime import datetime
import argparse

import yaml
import torch
import pandas as pd

from drugai.trainer import Trainer
from drugai.utils.common import override_defaults, seed_everything, default_collate_fn, load_dataset
from drugai import MODEL_CLASSES
from drugai.vocab import Vocab

logger = logging.getLogger(__file__)


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default=None, type=str, required=True, help="The parameter config path.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the dev set.")
    return parser


def main():
    args = get_argparse().parse_args()
    ## 修正的参数
    f = open(args.config_dir, "r")
    fix_args = yaml.full_load(f)
    for k in fix_args:
        args = override_defaults(args, fix_args[k])

    ## 随机种子
    seed_everything(args.seed)

    ## 模型输出目录
    args.output_dir = os.path.join(args.output_dir,
                                   args.model_name.lower() + "_" + args.data_name + "_" + str(args.seed))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## 日志输出
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    ## tensorboardx
    args.tensorboardx_path = os.path.join(args.tensorboardx_path, args.model_name.lower() + \
                                          "_" + args.data_name + "_" + str(args.seed) + \
                                          "_" + datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(args.tensorboardx_path):
        os.makedirs(args.tensorboardx_path)

    # Setup CUDA， GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        logger.info({"n_gpu: ": args.n_gpu})
    else:  # initializes the distributed backend which will take care of suchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_porcess_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    if args.do_train:
        train_dataset = load_dataset(args, "train")
        vocab = Vocab.from_data(train_dataset)
        args.vocab_size = len(vocab)
        args.padding_ids = vocab.pad_token_ids
        eval_dataset = load_dataset(args, "test")
        model = MODEL_CLASSES[args.model_name][0](args)

        collate_fn = default_collate_fn(vocab)
        trainer = Trainer(model=model, vocab=vocab, args=args, collate_fn=collate_fn, train_dataset=train_dataset,
                          eval_dataset=eval_dataset)
        trainer.train()

    if args.do_eval:
        pass

    if args.do_predict:
        pass


if __name__ == '__main__':
    main()
