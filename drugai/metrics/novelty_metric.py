#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:29
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : novelty_metric.py
from __future__ import annotations, print_function

import logging
import os
from typing import Optional, Dict, Text, Any, List, Tuple


from moses.metrics.metrics import novelty
from moses.script_utils import read_smiles_csv
from moses.utils import get_mol, mapper
from moses.metrics import remove_invalid
from rdkit.Chem.rdchem import Mol

from drugai.component import Component
from drugai.utils.io import read_smiles_zip

logger = logging.getLogger(__name__)


class NoveltyMetric(Component):
    defaults = {
        "train_dir": None
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(NoveltyMetric, self).__init__(component_config=component_config, **kwargs)

    def prepare_data(self,
                     filename: Text,
                     content: Dict,
                     **kwargs) -> Dict:
        if content.get(filename, None) is None:
            if content.get(filename, None) is None:
                if self.component_config[filename + "_dir"] is not None:
                    train_dir = self.component_config[filename + "_dir"]
                    train = read_smiles_csv(train_dir)
                else:
                    train_dir = os.path.join(os.path.dirname(__file__), "resources", filename + ".csv.gz")
                    train = read_smiles_zip(train_dir)
                content[filename] = train

        return content

    def train(self,
              smiles: List[Text, Mol],
              n_jobs=1,
              device: Text = "cpu",
              content: Dict = {},
              **kwargs: Any
              ) -> Tuple[Dict, Dict]:
        if content.get("train", None) is None:
            content = self.prepare_data(filename="train", content=content)
        similes = remove_invalid(smiles, canonize=True)  ## 移除无效的分子
        mols = mapper(n_jobs)(get_mol, similes)
        result = {'Novelty': novelty(gen=mols,
                                     train=content["train"],
                                     n_jobs=n_jobs)}
        logger.info("Novelty: %s" %(result["Novelty"]))
        return content, result
