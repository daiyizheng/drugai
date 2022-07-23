#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 21:06
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : model_training.py
from __future__ import annotations, print_function

import argparse
import logging

from drugai import Metric, Visualize
from drugai.models.interpreter import create_interpreter
from drugai.models.trainer import Trainer
from drugai.utils.io import read_config_yaml

logger = logging.getLogger(__name__)


def train(args: argparse.Namespace):
    config = read_config_yaml(args.config)
    print("no_cuda", args.no_cuda)
    trainer = Trainer(config,
                      tensorboardx_dir=args.tensorboardx_dir,
                      no_cuda=args.no_cuda,
                      local_rank=args.local_rank,
                      fp16=args.fp16,
                      fp16_opt_level=args.fp16_opt_level)
    interpreter = trainer.train(train_dir=args.train_dir, eval_dir=args.eval_dir)
    path = args.out
    trainer.persist(path)


def predict(args: argparse.Namespace):
    config = read_config_yaml(args.config)
    model_dir = "" if args.model is None else args.model
    interpreter = create_interpreter(cfg=config,
                                     model_dir=model_dir,
                                     no_cuda=args.no_cuda)
    results = interpreter.inference(test_dir=args.test_dir)
    interpreter.persist(args.out, results)


def visualize(args: argparse.Namespace)->None:
    vs = Visualize()
    vs.show(smiles1=args.smiles,
            save_file=args.save_file)


def metric(args: argparse.Namespace):
    config = read_config_yaml(args.config)
    met = Metric(config)
    content = met.compute(gen_dir=args.gen_dir,
                          n_jobs=args.num_workers,
                          device = "cpu" if args.no_cuda else "cuda")

    met.persist(path=args.out, content=content)
