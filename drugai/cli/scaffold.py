#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/7 9:10
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : scaffold.py
from __future__ import annotations, print_function
import argparse
from typing import List

from drugai.cli import SubParsersAction


def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]
                  ) -> None:
    """Add all test parsers."""
    scaffold_parser = subparsers.add_parser(
        "init",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Creates a new config files.",
    )
    scaffold_parser.set_defaults(func=scaffold_main)


def scaffold_main(args: argparse.Namespace) -> None:
    print("scaffold")