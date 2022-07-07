#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/3 23:36
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : drugai.py
from __future__ import annotations, print_function

import logging
import os
import argparse
from datetime import datetime
from functools import partial

import torch
import yaml
from rdkit.Chem import Draw
from rdkit import Chem

from drugai.models.dataset_base import default_collate_fn
from moses.script_utils import read_smiles_csv

from drugai.models import MODEL_CLASSES
from drugai.utils.common import override_defaults, seed_everything

logger = logging.getLogger(__name__)


class RunTask(object):

    @staticmethod
    def _load_config(config: argparse.Namespace):
        ## 修正的参数
        config_dir = config.config
        if config_dir is None:
            base_path = os.path.dirname(__file__)
            config_dir = os.path.join(base_path, "models", "model_arguments", config.model_name+".yml")
        if not os.path.exists(config_dir):
            logger.error(config.model_name+".yml"+" is not exist")
            raise KeyError
        f = open(config_dir, "r")
        fix_args = yaml.full_load(f)
        for k in fix_args:
            config = override_defaults(config, fix_args[k])
        return config


    def train(self, config: argparse.Namespace):
        ## 修正的参数
        config = self._load_config(config)

        ## 随机种子
        seed_everything(config.seed)

        ## 模型输出目录
        config.output_dir = os.path.join(config.output_dir, config.model_name.lower())
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        ## tensorboardx
        config.tensorboardx_dir = os.path.join(config.tensorboardx_dir, config.model_name.lower() + \
                                                "_" + datetime.now().strftime("%Y%m%d%H%M%S"))
        if not os.path.exists(config.tensorboardx_dir):
            os.makedirs(config.tensorboardx_dir)

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

        train_dataset = read_smiles_csv(config.train_dir)
        eval_dataset = read_smiles_csv(config.eval_dir)

        model_class, vocab_class, trainer_class, dataset_class = MODEL_CLASSES[config.model_name]
        vocab = vocab_class.from_data(train_dataset)
        config.vocab_size = len(vocab)
        print("vocab_size: ", len(vocab))

        model = model_class(vocab_size=config.vocab_size,
                            num_layers=config.num_layers,
                            dropout_rate=config.dropout_rate,
                            hidden_size=config.hidden_size,
                            pad_token_ids=vocab.pad_token_ids)
        collate_fn = partial(default_collate_fn, vocab)
        dataset = dataset_class(train_data=train_dataset,
                                eval_data=eval_dataset,
                                test_data=None,
                                batch_size=config.batch_size,
                                vocab=vocab,
                                num_workers=config.num_workers,
                                collate_fn=collate_fn)
        trainer = trainer_class(config)
        trainer.fit(model=model,
                    dataset=dataset,
                    vocab=vocab,
                    criterion=None)


    def sample(self, config: argparse.Namespace):
        pass

    def metric(self, config: argparse.Namespace):
        pass

    def visualize(self, config: argparse.Namespace):
        smiles = config.smiles
        mol = Chem.MolFromSmiles(smiles)

        if config.save_file:
            Draw.MolToFile(mol, config.save_file, size=(150, 150), fitImage=True, imageType='png')
            return
        Draw.MolToImage(mol, size=(150, 150), kekulize=True)





