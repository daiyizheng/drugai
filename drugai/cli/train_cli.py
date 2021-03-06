#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 21:08
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : train_cli.py
from __future__ import annotations, print_function

import logging
from typing import List
import argparse

from drugai.cli import SubParsersAction
from drugai.cli.default_arguments import train_default
from drugai.task import train
from drugai.utils.common import seed_everything

logger = logging.getLogger(__name__)


def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]
                  ) -> None:
    """Add all test parsers."""
    train_parser = subparsers.add_parser(
        "train",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="train molecular.",
    )
    train_parser.set_defaults(func=train_main)
    train_default.set_train_arguments(train_parser)


def train_main(args: argparse.Namespace) -> None:
    logger.info("train drugai start")
    ## 随机种子
    seed_everything(args.seed)
    train(args)
