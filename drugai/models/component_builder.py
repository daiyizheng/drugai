#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 22:40
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : component_builder.py
from __future__ import annotations, print_function

from typing import Dict, Text, Any

from drugai import registry


class ComponentBuilder:
    def __init__(self,):
        pass

    def load_component(self,
                       component_meta:Dict[Text, Any],
                       model_dir:Text,
                       **kwargs)-> "Component":
        return registry.load_component_by_meta(component_meta,  model_dir,  **kwargs)

    def create_component(self,
                         component_config: Dict[Text, Any],
                         **kwargs
                         )-> "Component":
        component = registry.create_component_by_config(component_config, **kwargs)
        return component

    def create_component_from_class(self, component_class):
        component_config = {"name": component_class.name}
        return self.create_component(component_config)