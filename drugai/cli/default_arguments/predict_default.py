#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 20:43
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : sample_default.py
from __future__ import annotations, print_function

import argparse
from typing import Union

from drugai.cli.default_arguments.default import add_output_param, add_model_param, add_init_param, add_config_param, \
    add_distributed_param, add_parallel_param, add_f16_param, add_device_param
from drugai.utils.constants import DEFAULT_RESULTS_PATH


def add_input_test_data_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
                              ) -> None:
    parser.add_argument(
        "--test_dir",
        type=str,
        required=False,
        help="",
    )

def set_predict_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets the CLI arguments for `drugai predict."""
    add_init_param(parser)
    add_config_param(parser)
    add_input_test_data_param(parser)
    add_output_param(parser, default=DEFAULT_RESULTS_PATH)
    add_model_param(parser)
    add_distributed_param(parser)
    add_parallel_param(parser)
    add_f16_param(parser)
    add_device_param(parser)
