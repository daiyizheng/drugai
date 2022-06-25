#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/25 20:08
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : component.py
from __future__ import annotations

import abc


class Component(abc.ABC):
    def __init__(self):
        self.component_name = Component.__name__