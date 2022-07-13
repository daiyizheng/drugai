#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 20:45
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : visualize_default.py
from __future__ import annotations, print_function

import argparse


def add_out_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        help="",
    )


def add_input_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="",
    )


def add_draw_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--size",
        type=tuple,
        required=False,
        default=(200,200),
        help="",
    )
    parser.add_argument(
        "--legend",
        type=str,
        required=False,
        default=None,
        help="",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default="signle",
        help="",
    )
    parser.add_argument(
        "--molsPerRow",
        type=int,
        required=False,
        default=4,
        help="",
    )
    parser.add_argument(
        "--subImgSize",
        type=tuple,
        required=False,
        default=(200,200),
        help="",
    )

    parser.add_argument(
        "--legends",
        type=list,
        required=False,
        default=[],
        help="",
    )


def set_visualize_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets the CLI arguments for `drugai data visualize."""
    add_out_param(parser)
    add_input_param(parser)
    add_draw_param(parser)
