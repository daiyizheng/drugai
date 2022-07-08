#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/7 9:10
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : scaffold.py
from __future__ import annotations, print_function
import argparse
import os
import sys
from typing import List, Text

from drugai.cli import SubParsersAction
from drugai.cli.default_arguments import scaffold_default
from drugai.utils.constants import AUTHOR_EMAIL


def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]
                  ) -> None:
    """Add all test parsers."""
    scaffold_parser = subparsers.add_parser(
        "init",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Creates a new config files.",
    )
    scaffold_parser.add_argument(
        "--init_dir",
        default=None,
        help="Directory where your project should be initialized.",
    )
    scaffold_default.set_scaffold_arguments(scaffold_parser)
    scaffold_parser.set_defaults(func=scaffold_main)


def _ask_create_path(path: Text) -> None:
    import questionary

    should_create = questionary.confirm(
        f"Path '{path}' does not exist 🧐. Create path?"
    ).ask()

    if should_create:
        try:
            os.makedirs(path)
        except (PermissionError, OSError, FileExistsError) as e:
            print(
                f"\033[91m Failed to create project path at '{path}'. " f"Error: {e} \033[0m"
            )
    else:
        print(
            "\033[92m Ok, will exit for now. You can continue setting up by  running 'DrugAi init' again 🙋🏽‍♀️"
        )
        sys.exit(0)


def create_initial_project(path: Text) -> None:
    """Creates directory structure and templates for initial project."""
    from distutils.dir_util import copy_tree

    copy_tree(scaffold_path(), path)


def scaffold_path() -> Text:
    import pkg_resources

    return pkg_resources.resource_filename(__name__, "initial_project")


def init_project(path: Text) -> None:
    os.chdir(path)
    create_initial_project(".")
    print(f"Created project directory at '{os.getcwd()}'.")


def scaffold_main(args: argparse.Namespace
                  ) -> None:

    import questionary
    print("\033[92m Welcome to DrugAi! 🤖\n \033[0m")

    print(
        f"To get started quickly, an "
        f"initial project will be created.\n"
        f"If you need some help, check out "
        f"the Email at {AUTHOR_EMAIL}.\n"
        f"Now let's start! 👇🏽\n"
    )

    if args.init_dir is not None:
        path = args.init_dir
    else:
        path = (
            questionary.text(
                "Please enter a path where the project will be "
                "created [default: current directory]"
            ).ask()
        )

    if path == "":
        path = "."

    if path is None or not os.path.isdir(path):
        print("\033[92m Ok. You can continue setting up by running 'DrugAi init' 🙋🏽‍♀️\033[0m")
        sys.exit(0)

    init_project(path)
