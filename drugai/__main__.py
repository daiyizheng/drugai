#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 17:00
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __main__.py
from __future__ import annotations, print_function

import logging
import os
import sys
sys.path.insert(0,"./")
from datetime import datetime
from colorama import Fore, Back, Style
import argparse

from drugai.cli import (
    train_cli,
    sample_cli,
    metric_cli,
    visualize_cli,
    scaffold_cli)
from drugai.cli.default_arguments.default import add_logging_options
from drugai.utils.common import configure_colored_logging, configure_logging_and_warnings
from drugai.version import __version__

dt = datetime.now()

__VERSION__ = f'👍    {__version__}'
__AUTHOR__ = '😀    Yizheng Dai'
__CONTACT__ = '😍    qq: 387942239'
__DATE__ = f"👉    {dt.strftime('%Y.%m.%d')}, since 2022.06.11"
__LOC__ = '👉    Hangzhou, China'
__git__ = '👍    https://github.com/daiyizheng/drugai'

logger = logging.getLogger(__file__)

def create_arg_parse():
    """
    parse arguments
    :return:
    """

    parser = argparse.ArgumentParser(prog="drugai",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="DrugAi command line Tools. Rasa allows you to build")
    parser.add_argument( "--version",
                         action="store_true",
                         default=argparse.SUPPRESS,
                         help="Print installed drugai version",)
    # train, sample, metric, vision
    parent_parser = argparse.ArgumentParser(add_help=False)
    #logging level
    add_logging_options(parent_parser)
    parent_parsers = [parent_parser]

    subparsers = parser.add_subparsers(help="DrugAi commands")

    train_cli.add_subparser(subparsers, parents=parent_parsers)
    sample_cli.add_subparser(subparsers, parents=parent_parsers)
    metric_cli.add_subparser(subparsers, parents=parent_parsers)
    visualize_cli.add_subparser(subparsers, parents=parent_parsers)
    scaffold_cli.add_subparser(subparsers, parents=parent_parsers)
    return parser


def print_version():
    print('-' * 70)
    print(Fore.BLUE + Style.BRIGHT + '              Alfred ' + Style.RESET_ALL +
          Fore.WHITE + '- Valet of Artificial Intelligence.' + Style.RESET_ALL)
    print('         Author : ' + Fore.CYAN +
          Style.BRIGHT + __AUTHOR__ + Style.RESET_ALL)
    print('         Contact: ' + Fore.BLUE +
          Style.BRIGHT + __CONTACT__ + Style.RESET_ALL)
    print('         At     : ' + Fore.LIGHTGREEN_EX +
          Style.BRIGHT + __DATE__ + Style.RESET_ALL)
    print('         Loc    : ' + Fore.LIGHTMAGENTA_EX +
          Style.BRIGHT + __LOC__ + Style.RESET_ALL)
    print('         Star   : ' + Fore.MAGENTA +
          Style.BRIGHT + __git__ + Style.RESET_ALL)
    print('         Ver.   : ' + Fore.GREEN +
          Style.BRIGHT + __VERSION__ + Style.RESET_ALL)
    print('-' * 70)
    print('\n')


def main():
    arg_parser = create_arg_parse()
    cmdline_arguments = arg_parser.parse_args()


    try:
        if hasattr(cmdline_arguments, "func"):

            log_level = getattr(cmdline_arguments, "loglevel", None)
            configure_logging_and_warnings(log_level,
                                           warn_only_once=False,
                                           filter_repeated_logs=False)
            configure_colored_logging(log_level)
            cmdline_arguments.func(cmdline_arguments)
        elif hasattr(cmdline_arguments, "version"):
            print_version()
        else:
            # user has not provided a subcommand, let's print the help
            logger.error("No command specified.")
            arg_parser.print_help()
            sys.exit(1)
    except Exception as e:
        # these are exceptions we expect to happen (e.g. invalid training data format)
        # it doesn't make sense to print a stacktrace for these if we are not in
        # debug mode
        logger.debug("Failed to run CLI command due to an exception.", exc_info=e)
        sys.exit(1)

if __name__ == '__main__':
    main()
