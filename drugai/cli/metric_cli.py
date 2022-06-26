#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 21:09
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : metric_cli.py
from __future__ import annotations, print_function
import argparse
from typing import List

from drugai.cli import SubParsersAction


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


def metric_main(args: argparse.Namespace) -> None:
    print("visualize")