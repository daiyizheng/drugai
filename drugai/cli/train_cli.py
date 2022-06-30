#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 21:08
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : train_cli.py
from __future__ import annotations, print_function

from typing import List
import argparse

from drugai.cli import SubParsersAction


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


def train_main(args: argparse.Namespace) -> None:
    print("train")