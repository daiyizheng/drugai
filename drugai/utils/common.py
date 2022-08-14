#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 13:46
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : common.py
from __future__ import annotations, print_function
from ast import List

import importlib
from typing import Callable, Optional, Any, Text, Dict
import copy
import logging
import os
import random
import sys
import argparse

import numpy as np
import torch
from rdkit.Chem.rdchem import Mol

from drugai.utils.constants import ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def override_defaults(
        defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """
    Returns:
        updated config
    """
    if defaults:
        config = copy.deepcopy(defaults)
    else:
        config = {}

    if custom:
        for key in custom.keys():
            if isinstance(config, argparse.ArgumentParser):
                setattr(config, key, custom[key])
            else:
                if isinstance(config.get(key), dict):
                    config[key].update(custom[key])
                else:
                    config[key] = custom[key]
    return config


def configure_logging_and_warnings(log_level: Optional[int] = None,
                                   warn_only_once: bool = False,
                                   filter_repeated_logs: bool = False
                                   ) -> None:
    if log_level is None:
        log_level_name = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
        log_level = logging.getLevelName(log_level_name)

    logging.getLogger("drugai").setLevel(log_level)

    if filter_repeated_logs:  # 过滤重复
        raise NotImplementedError

    if warn_only_once:  # 过滤警告
        raise NotImplementedError


def configure_colored_logging(loglevel: Text) -> None:
    """Configures coloredlogs library for specified loglevel.
    """
    import coloredlogs

    loglevel = loglevel or os.environ.get(
        ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL
    )

    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles["debug"] = {}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        fmt="%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
        level_styles=level_styles,
        field_styles=field_styles,
    )


def class_from_module_path(
        module_path: Text, lookup_path: Optional[Text] = None
) -> Any:
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects."""
    # load the module, will raise ImportError if module cannot be loaded
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition(".")
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        return getattr(m, class_name)
    else:
        module = globals().get(module_path, locals().get(module_path))
        if module is not None:
            return module

        if lookup_path:
            # last resort: try to import the class from the lookup path
            m = importlib.import_module(lookup_path)
            return getattr(m, module_path)
        else:
            raise ImportError(f"Cannot retrieve class from path {module_path}.")

def minimal_kwargs(
    kwargs: Dict[Text, Any], func: Callable, excluded_keys: Optional[List] = None
) -> Dict[Text, Any]:
    """Returns only the kwargs which are required by a function. Keys, contained in
    the exception list, are not included.
    """

    excluded_keys = excluded_keys or []

    possible_arguments = arguments_of(func)

    return {
        k: v
        for k, v in kwargs.items()
        if k in possible_arguments and k not in excluded_keys
    }

def arguments_of(func: Callable) -> List[Text]:
    """Return the parameters of the function `func` as a list of names."""
    import inspect

    return list(inspect.signature(func).parameters.keys())

def type_check_num_atoms(mol: Mol, 
                         num_max_atoms:int=-1):
    """Check number of atoms in `mol` does not exceed `num_max_atoms`

    If number of atoms in `mol` exceeds the number `num_max_atoms`, it will
    raise `KeyError` exception.

    Args:
        mol (Mol):
        num_max_atoms (int): If negative value is set, not check number of
            atoms.

    """
    num_atoms = mol.GetNumAtoms()
    if num_max_atoms >= 0 and num_atoms > num_max_atoms:
        # Skip extracting feature. ignore this case.
        
        raise KeyError(
            'Number of atoms in mol {} exceeds num_max_atoms {}'
            .format(num_atoms, num_max_atoms))