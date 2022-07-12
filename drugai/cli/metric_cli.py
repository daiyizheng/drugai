#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 21:09
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : metric_cli.py
from __future__ import annotations, print_function
import argparse
import logging
from typing import List

from drugai.cli import SubParsersAction
from drugai.cli.default_arguments import metric_default
from drugai.task import metric

logger = logging.getLogger(__name__)


def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]
                  ) -> None:
    """Add all test parsers."""
    metric_parser = subparsers.add_parser(
        "metric",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="metric molecular evaluate.",
    )
    metric_parser.set_defaults(func=metric_main)
    metric_default.set_metric_arguments(metric_parser)


def metric_main(args: argparse.Namespace) -> None:
    logger.info("metric start")
    logger.info("args:{}".format(vars(args)))
    metric(args)
