#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 13:59
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trainer.py
from __future__ import annotations, print_function

import argparse
import os
from datetime import datetime
from typing import Text, Optional, Dict, Any

from drugai.models.component_builder import ComponentBuilder
from drugai.models.interpreter import Interpreter
from drugai.utils.io import create_directory


class Trainer(object):
    def __init__(self, cfg: Dict[Text, Any],
                 **kwargs):
        self.config = cfg
        component_builder = ComponentBuilder()
        self.pipeline = self._build_pipeline(cfg, component_builder, **kwargs)

    def persist(
            self,
            path: Text
    ) -> Text:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.abspath(path)
        # model_name = self.pipeline.name
        # dir_name = os.path.join(path, model_name)
        create_directory(path)

        self.pipeline.persist(path)
        return ""

    def _build_pipeline(self,
                        cfg: Dict[Text, Any],
                        component_builder,
                        **kwargs):
        component_cfg = cfg["train"][0]
        pipeline = component_builder.create_component(component_cfg, **kwargs)

        return pipeline

    def train(self,
              args: argparse.Namespace
              )->"Interpreter":
        self.pipeline.train(args.train_dir, args.eval_dir)

        return Interpreter(self.pipeline)
