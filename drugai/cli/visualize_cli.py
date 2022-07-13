#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 21:09
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : vis_cli.py
from __future__ import annotations, print_function

from typing import List
import argparse
import logging

from drugai.cli import SubParsersAction
from drugai.cli.default_arguments import visualize_default
from drugai.task import visualize

logger = logging.getLogger(__name__)

def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]
                  ) -> None:
    """Add all test parsers."""
    visualize_parser = subparsers.add_parser(
        "visualize",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Visualize molecular image.",
    )
    visualize_parser.set_defaults(func=visualize_main)
    visualize_default.set_visualize_arguments(visualize_parser)


def visualize_main(args: argparse.Namespace) -> None:
    logger.info("visualize satrt")
    visualize(args)

