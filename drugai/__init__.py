#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 20:51
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py
from __future__ import annotations, print_function

import logging

from drugai import version


__version__ = version.__version__


logging.getLogger(__name__).addHandler(logging.NullHandler())