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
                      add_device_param
                      )


def add_input_data_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
                         ) -> None:
    parser.add_argument(
        "-d",
        "--data_name",
        type=str,
        default="UNKNOW",
        help="",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=False,
        help="",
    )


def add_train_base_parm(parser: argparse.ArgumentParser
                        ) -> None:
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="",
    )

    parser.add_argument(
        "--evaluate_during_training",
        type=bool,
        default=True,
        help="",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
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
    add_input_data_param(parser)
    add_config_param(parser)
    add_train_base_parm(parser)
