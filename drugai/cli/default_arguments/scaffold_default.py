#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/7 9:10
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : scaffold_default.py
from __future__ import annotations, print_function

import argparse


def set_scaffold_arguments(parser: argparse.ArgumentParser
                           )->None:
    parser.add_argument(
        "--init_dir",
        default=None,
        help="Directory where your project should be initialized.",
    )