#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 21:06
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : model_training.py
from __future__ import annotations, print_function

import argparse
import os

from rdkit.Chem import Draw
from rdkit import Chem

from drugai.models.interpreter import create_interpreter
from drugai.models.trainer import Trainer
from drugai.utils.io import read_config_yaml, create_directory


def train(args: argparse.Namespace):
    config = read_config_yaml(args.config)
    trainer = Trainer(config,
                      tensorboardx_dir=args.tensorboardx_dir,
                      no_cuda=args.no_cuda,
                      local_rank=args.local_rank,
                      fp16=args.fp16,
                      fp16_opt_level=args.fp16_opt_level)
    interpreter = trainer.train(args)
    path = args.out
    trainer.persist(path)


def predict(args: argparse.Namespace):
    config = read_config_yaml(args.config)
    model_dir =  ""  if args.model is None else args.model
    interpreter = create_interpreter(cfg=config, model_dir=model_dir)
    results = interpreter.inference(test_dir=args.test_dir)
    interpreter.persist(args.out, results)


def visualize(args: argparse.Namespace):
    smiles = args.smiles
    mol = Chem.MolFromSmiles(smiles)

    if args.save_file:
        Draw.MolToFile(mol, args.save_file, size=(150, 150), fitImage=True, imageType='png')
        return
    Draw.MolToImage(mol, size=(150, 150), kekulize=True)


def metric(args: argparse.Namespace):
    pass
