#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:29
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : novelty_metric.py
from __future__ import annotations, print_function

import os
from typing import Optional, Dict, Text, Any, List, Tuple

import pandas as pd

from drugai.utils.io import read_smiles_zip
from moses.metrics.metrics import novelty

from moses.script_utils import read_smiles_csv
from moses.utils import get_mol, mapper
from moses.metrics import remove_invalid
from rdkit.Chem.rdchem import Mol

from drugai.component import Component


class NoveltyMetric(Component):
    defaults = {
        "train_dir":None
    }

    def __init__(self,
                 cfg: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(NoveltyMetric, self).__init__(component_config=cfg, **kwargs)

    def prepare_data(self,
                     filename: Text,
                     content: Dict,
                     **kwargs)->Dict:
        if content.get(filename, None) is None:
            if content.get(filename, None) is None:
                if self.component_config[filename + "_dir"] is None:
                    test_dir = self.component_config[filename + "_dir"]
                    test = read_smiles_csv(test_dir)
                else:
                    test_dir = os.path.join(os.path.dirname(__file__), "resources", filename + ".csv.gz")
                    test = read_smiles_zip(test_dir)
                content[filename] = test

        return content

    def train(self,
              similes: List[Text, Mol],
              n_jobs=1,
              device:Text="cpu",
              content:Dict={},
              **kwargs: Any
              ) -> Tuple[Dict, Dict]:
        if content.get("train", None) is None:
            content = self.prepare_data(filename="train", content=content)
        similes = remove_invalid(similes, canonize=True)  ## 移除无效的分子
        mols = mapper(n_jobs)(get_mol, similes)
        return content, {'Novelty' : novelty(gen=mols,
                                    train=content["train"],
                                    n_jobs=n_jobs)}