#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 19:47
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : default.py
# code from rasa-2.4.0
from __future__ import annotations, print_function

import argparse
import logging
from typing import Optional, Text

from drugai.utils.constants import DEFAULT_MODELS_PATH


def add_init_param(parser: argparse.ArgumentParser
                   ) -> None:
    parser.add_argument(
        "-ss",
        "--seed",
        type=int,
        default=1314,
        help="",
    )


def add_model_param(parser: argparse.ArgumentParser
                    ) -> None:
    parser.add_argument(
        "-mn",
        "--model",
        type=str,
        required=False,
        default="models/",
        help="",
    )


def add_config_param(parser: argparse.ArgumentParser
                     ) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=None,
        help=""
    )


def add_output_param(parser: argparse.ArgumentParser,
                     default: Optional[Text] = DEFAULT_MODELS_PATH,
                     ) -> None:
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=default,
        help=""
    )


def add_tensorboardx_log_param(parser: argparse.ArgumentParser
                               ) -> None:
    parser.add_argument(
        "--tensorboardx_dir",
        type=str,
        required=False,
        default="experiments/runs",
        help=""
    )


def add_distributed_param(parser: argparse.ArgumentParser
                          ) -> None:
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="",
    )


def add_parallel_param(parser: argparse.ArgumentParser
                       ) -> None:
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="",
    )


def add_f16_param(parser: argparse.ArgumentParser
                  ) -> None:
    parser.add_argument(
        "--fp16",
        type=bool,
        default=False,
        help="",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        choices=['00', '01', '02', '03'],
        default='01',
        help="",
    )


def add_device_param(parser: argparse.ArgumentParser
                     ) -> None:
    parser.add_argument(
        "--no_cuda",
        type=bool,
        default=False,
        help="",
    )


def add_logging_options(parser: argparse.ArgumentParser) -> None:
    """Add options to an argument parser to configure logging levels."""
    logging_arguments = parser.add_argument_group(
        "Python Logging Options",
        "You can control level of log messages printed. "
        "In addition to these arguments, a more fine grained configuration can be "
        "achieved with environment variables. See online documentation for more info.",
    )

    # arguments for logging configuration
    logging_arguments.add_argument(
        "-v",
        "--verbose",
        help="Be verbose. Sets logging level to INFO.",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    logging_arguments.add_argument(
        "-vv",
        "--debug",
        help="Print lots of debugging statements. Sets logging level to DEBUG.",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
    )

    logging_arguments.add_argument(
        "--quiet",
        help="Be quiet! Sets logging level to WARNING.",
        action="store_const",
        dest="loglevel",
        const=logging.WARNING,
    )
