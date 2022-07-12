#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 21:09
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : sample_cli.py
from __future__ import annotations, print_function
import argparse
from typing import List
import logging

from drugai.cli import SubParsersAction
from drugai.cli.default_arguments import predict_default
from drugai.task import predict
from drugai.utils.common import seed_everything

logger = logging.getLogger(__name__)


def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]
                  ) -> None:
    """Add all test parsers."""
    predict_parser = subparsers.add_parser(
        "predict",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="sample molecular .",
    )
    predict_parser.set_defaults(func=predict_main)
    predict_default.set_predict_arguments(predict_parser)


def predict_main(args: argparse.Namespace) -> None:
    logger.info("predict drugai start")
    ## 随机种子
    seed_everything(args.seed)
    predict(args)
