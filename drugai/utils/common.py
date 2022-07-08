#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 13:46
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : common.py
from __future__ import annotations, print_function

import copy
import logging
import os
import random
import sys
from typing import Optional, Any, Text, Dict

import numpy as np
import torch

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
            setattr(config, key, custom[key])

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

