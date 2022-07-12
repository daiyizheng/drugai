#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/10 21:12
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : interpreter.py
from __future__ import annotations, print_function

import os
from typing import Text, Optional, Dict, Any

import torch

from drugai.models.component import Component
from drugai.models.component_builder import ComponentBuilder
from drugai.utils.common import override_defaults
from drugai.utils.io import create_directory, write_dict_to_csv


def create_interpreter(cfg: Dict[Text, Any], model_dir: Text, **kwargs):
    return Interpreter.load(cfg, model_dir, **kwargs)


class Interpreter(object):
    def __init__(self,
                 pipeline: Component,
                 **kwargs):
        self.pipeline: Component = pipeline

    def inference(self, *args, **kwargs):
        return self.pipeline.process(*args, **kwargs)

    def persist(self, path:Text, content:Dict):
        path = os.path.abspath(path)
        model_name = self.pipeline.name
        create_directory(path)
        write_dict_to_csv(content,
                          file_path=os.path.join(path, model_name+"_results.csv"))

    @staticmethod
    def load(cfg:Dict[Text,Any],
             model_dir: Text,
             component_builder: Optional[ComponentBuilder] = None,
             **kwargs
             ) -> "Interpreter":
        new_config = cfg["predict"][0]
        return Interpreter.create(model_dir, component_builder, new_config=new_config, **kwargs)

    @staticmethod
    def create(model_dir: Text,
               component_builder: Optional[ComponentBuilder] = None,
               new_config: Optional[Dict] = None,
               **kwargs
               ) -> "Interpreter":
        if component_builder is None:
            component_builder = ComponentBuilder()
        component_meta = torch.load(os.path.join(model_dir, new_config["name"] + "_component_config.pt"))
        if new_config is not None:
            if new_config["name"] != component_meta["name"]:
                raise KeyError
            component_meta = override_defaults(component_meta, new_config)
        pipeline = component_builder.load_component(component_meta, model_dir, **kwargs)
        return Interpreter(pipeline)
