#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 21:05
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : generate.py
from __future__ import annotations, print_function

import logging
import os
import argparse

import torch
import pandas as pd
import yaml
from tqdm import tqdm

from drugai import MODEL_CLASSES
from drugai.utils.common import override_defaults

logger = logging.getLogger(__file__)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_config_dir", default=None, type=str, required=True, help = "The parameter config path for samples.")

    return parser


def main():
    args = get_argparse().parse_args()
    ## 修正的参数
    f = open(args.generate_config_dir, "r")
    fix_args = yaml.full_load(f)
    for k in fix_args:
        args = override_defaults(args, fix_args[k])

    ## 是否使用显卡
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

    if not os.path.exists(os.path.join(args.param_dir, "vocab.pt")):
        raise KeyError("vocab path error")
    vocab = torch.load(os.path.join(args.param_dir, "vocab.pt"))
    args.vocab_size = len(vocab)
    if not os.path.exists(os.path.join(args.param_dir, "model.pt")):
        raise KeyError("model path error")

    model = MODEL_CLASSES[args.model_name][0](args=args)
    model.load_state_dict(torch.load(os.path.join(args.param_dir, "model.pt")))
    trainer = MODEL_CLASSES[args.model_name][2](model=model,
                                                vocab=vocab,
                                                args=args)
    gen_number = args.gen_number
    samples = []
    with tqdm(args.gen_number, desc="Generating sample") as T:
        while gen_number > 0:
            current_sample = trainer.predict(gen_number)
            samples.extend(current_sample)
            gen_number -= len(current_sample)
            T.update(len(current_sample))
    samples = pd.DataFrame(samples, columns=['SMILES'])
    samples.to_csv(os.path.join(args.output_dir, "smiles.csv"), index=False, encoding="utf_8_sig")

if __name__ == '__main__':
    main()




