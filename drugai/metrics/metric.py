#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 9:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : metric.py
from __future__ import annotations, print_function

import os
from pathlib import Path
from typing import List, Text, Dict, Any, Optional
import logging

from moses.script_utils import read_smiles_csv
from rdkit.Chem.rdchem import Mol

from drugai.models.component_builder import ComponentBuilder
from drugai.utils.io import create_directory, write_json_to_file

logger = logging.getLogger(__name__)


class Metric(object):
    def __init__(self,
                 cfg: Dict[Text, Any],
                 **kwargs):
        self.cfg = cfg
        component_builder = ComponentBuilder()
        self.metics_pipeline = self._build_metrics(cfg=cfg,
                                                   component_builder=component_builder,
                                                   **kwargs)
    @staticmethod
    def prepare_data(path: Optional[Text, Path]):

        return read_smiles_csv(path)

    def compute(self,
                gen_dir: List[Optional[Text, Mol]],
                **kwargs):
        similes = self.prepare_data(gen_dir)
        metric_results = {}
        for i, component in enumerate(self.metics_pipeline):
            logger.info(f"Starting to train component {component.name}")
            content = component.train(similes, **kwargs)
            if content:
                metric_results.update(content)
            logger.info("Finished training component.")
        return metric_results

    def persist(self,
                path: Optional[Text, Path],
                content: Dict[Text, Any],
                **kwargs):
        path = os.path.abspath(path)
        create_directory(path)
        write_json_to_file(filename=os.path.join(path, "metric.csv"),
                           obj=content)

    def _build_metrics(self,
                       cfg: Dict[Text, Any],
                       component_builder: "ComponentBuilder",
                       **kwargs):
        metics_pipeline = []
        component_cfgs = cfg["metric"]
        for index, pipeline_component in enumerate(component_cfgs):
            component = component_builder.create_component(pipeline_component, **kwargs)
            metics_pipeline.append(component)
        return metics_pipeline
