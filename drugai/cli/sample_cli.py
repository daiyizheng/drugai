#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 21:09
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : sample_cli.py
from __future__ import annotations, print_function
import argparse
from typing import List

from drugai.cli import SubParsersAction


def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]
                  ) -> None:
    """Add all test parsers."""
    sample_parser = subparsers.add_parser(
        "sample",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="sample molecular .",
    )
    sample_parser.set_defaults(func=sample_main)


def sample_main(args: argparse.Namespace) -> None:
    print("sample")