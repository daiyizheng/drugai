#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/10 21:12
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : interpreter.py
from __future__ import annotations, print_function

import os
from typing import Text, Optional, Dict

import torch

from drugai.models.component import Component
from drugai.models.component_builder import ComponentBuilder
from drugai.utils.common import override_defaults


def create_interpreter(model_dir:Text, **kwargs):
    return Interpreter.load(model_dir, **kwargs)


class Interpreter(object):
    def __init__(self,
                 pipeline:Component,
                 **kwargs):
        self.pipeline:Component = pipeline

    def inference(self, *args, **kwargs):
        return self.pipeline.predict(*args, **kwargs)


    @staticmethod
    def load(model_dir:Text,
             component_builder: Optional[ComponentBuilder] = None,
             **kwargs
             )-> "Interpreter":
        return Interpreter.create(model_dir, component_builder, **kwargs)

    @staticmethod
    def create(model_dir: Text,
               component_builder: Optional[ComponentBuilder] = None,
               new_config: Optional[Dict] = None,
               **kwargs
               )-> "Interpreter":
        if component_builder is None:
            component_builder = ComponentBuilder()
        component_meta = torch.load(os.path.join(model_dir, "component_config.pt"))
        if new_config is not None:
            if new_config["name"] == component_meta["name"]:
                component_meta = override_defaults(component_meta, new_config)
        pipeline = component_builder.load_component(component_meta, model_dir, **kwargs)
        return Interpreter(pipeline)

