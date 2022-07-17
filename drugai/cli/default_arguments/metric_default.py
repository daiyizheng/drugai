#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 20:45
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : metric_default.py
from __future__ import annotations, print_function
import argparse

from drugai.cli.default_arguments.default import add_device_param, add_output_param, add_config_param, \
    add_parallel_param
from drugai.utils.constants import DEFAULT_RESULTS_PATH, DEFAULT_N_JOBS


def add_gen_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gen_dir",
        type=str,
        default=True,
        help="",
    )


def set_metric_arguments(parser: argparse.ArgumentParser
                         ) -> None:
    """Sets the CLI arguments for `drugai metric."""
    add_device_param(parser)
    add_gen_param(parser)
    add_output_param(parser, default=DEFAULT_RESULTS_PATH)
    add_parallel_param(parser, default=DEFAULT_N_JOBS)
    add_config_param(parser)

