#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/25 20:32
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : samples.py
from __future__ import annotations, print_function
import argparse
import logging
import os

import torch
import yaml
import pandas as pd

from drugai import MODEL_CLASSES
from drugai.utils.common import override_defaults, seed_everything

logger = logging.getLogger(__file__)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default=None, type=str, required=False,
                        help = "The parameter config path for samples.")
    parser.add_argument("--is_load_save_config", default=True, type=bool, required=True,
                        help="is load train config")
    parser.add_argument("--load_save_config", default=None, type=str, required=False,
                        help="train config path")
    parser.add_argument("--sample_dir", default=None, type=str, required=True,
                        help="sample path save path")

    return parser

def main():
    config = get_argparse().parse_args()
    if config.is_load_save_config:
        if config.load_save_config is None:
            raise KeyError
        load_config = torch.load(os.path.join(config.load_save_config, "args.pt"))
        config = override_defaults(config, vars(load_config))
    else:
        if config.config_dir is None:
            raise KeyError
        f = open(config.config_dir, "r")
        load_config = yaml.full_load(f)
        for k in load_config:
            config = override_defaults(config, load_config[k])

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

    model_class, _, trainer_class, dataset_class = MODEL_CLASSES[config.model_name]
    vocab = torch.load(os.path.join(os.path.join(config.load_save_config, "vocab.pt")))
    config.vocab_size = len(vocab)
    model = model_class(vocab_size=config.vocab_size,
                        num_layers=config.num_layers,
                        dropout_rate=config.dropout_rate,
                        hidden_size=config.hidden_size,
                        pad_token_ids=vocab.pad_token_ids)
    model.load_state_dict(torch.load(os.path.join(config.load_save_config, "model.pt")))

    dataset = dataset_class(train_data=None,
                            eval_data=None,
                            test_data=None,
                            batch_size=config.batch_size,
                            vocab=vocab,
                            num_workers=config.num_workers,
                            collate_fn=None)
    trainer = trainer_class(config)
    samples= trainer.sample(n_sample=config.n_sample,
                            model=model,
                            dataset=dataset,
                            vocab=vocab)
    samples = pd.DataFrame(samples, columns=['SMILES'])
    samples.to_csv(os.path.join(config.sample_dir, config.model_name, "smiles.csv"), index=False, encoding="utf_8_sig")

if __name__ == '__main__':
    main()

