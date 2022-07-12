#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 20:43
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : train_default.py
from __future__ import annotations, print_function

from typing import Union
import argparse

from .default import (add_init_param,
                      add_model_param,
                      add_config_param,
                      add_output_param,
                      add_tensorboardx_log_param,
                      add_distributed_param,
                      add_parallel_param,
                      add_f16_param,
                      add_device_param)


def add_input_train_data_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
                               ) -> None:
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="",
    )


def add_input_eval_data_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
                              ) -> None:
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        required=False,
        help="",
    )


def add_data_name_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
                        ) -> None:
    parser.add_argument(
        "--data_name",
        type=str,
        default="UNKNOW",
        help="",
    )


def set_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets the CLI arguments for `drugai data visualize."""
    add_init_param(parser)
    add_output_param(parser)
    add_tensorboardx_log_param(parser)
    add_distributed_param(parser)
    add_parallel_param(parser)
    add_f16_param(parser)
    add_device_param(parser)
    add_model_param(parser)
    add_input_train_data_param(parser)
    add_input_eval_data_param(parser)
    add_config_param(parser)
