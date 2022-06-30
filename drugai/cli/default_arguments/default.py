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