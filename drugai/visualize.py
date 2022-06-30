#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/22 23:08
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : visualize.py
from __future__ import annotations, print_function
import argparse



def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default=None, type=str, required=True,
                        help="The parameter config path for samples.")
    return parser


def main():
    print("可视化未完成")
