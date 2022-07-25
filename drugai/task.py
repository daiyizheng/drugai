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
from drugai.shared.importers.drug_importer import DrugImporter
from drugai.utils.io import read_config_yaml

logger = logging.getLogger(__name__)


def train(args: argparse.Namespace):
    file_importer = DrugImporter(config_file=args.config,
                                 train_data_paths=args.train_dir,
                                 eval_data_paths=args.eval_dir)
    print("no_cuda", args.no_cuda)
    trainer = Trainer(file_importer=file_importer,
                      tensorboardx_dir=args.tensorboardx_dir,
                      no_cuda=args.no_cuda,
                      local_rank=args.local_rank,
                      fp16=args.fp16,
                      fp16_opt_level=args.fp16_opt_level)
    interpreter = trainer.train()
    path = args.out
    trainer.persist(path)


def predict(args: argparse.Namespace):
    file_importer = DrugImporter(config_file=args.config,
                                 test_data_paths=args.test_dir)
    print("no_cuda", args.no_cuda)
    model_dir = "" if args.model is None else args.model
    interpreter = create_interpreter(cfg=file_importer.get_config(),
                                     model_dir=model_dir,
                                     no_cuda=args.no_cuda)
    results = interpreter.inference(file_importer = file_importer)
    interpreter.persist(args.out, results)


def visualize(args: argparse.Namespace)->None:
    vs = Visualize()
    vs.show(smiles1=args.smiles,
            save_file=args.save_file)


def metric(args: argparse.Namespace):
    file_importer = DrugImporter(config_file=args.config,
                                 test_data_paths=args.test_dir)
    print("no_cuda", args.no_cuda)
    met = Metric(file_importer.get_config())
    content = met.compute(gen_dir=args.gen_dir,
                          n_jobs=args.num_workers,
                          device = "cpu" if args.no_cuda else "cuda")

    met.persist(path=args.out, content=content)
