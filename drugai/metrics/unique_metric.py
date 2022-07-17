#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 13:20
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : unique_metric.py
from __future__ import annotations, print_function

from typing import Optional, Dict, Text, Any, List, Tuple

from moses.metrics import fraction_unique, remove_invalid
from rdkit.Chem.rdchem import Mol

from drugai.component import Component


class UniqueMetic(Component):
    defaults = {
        "unique_k": [1000, 10000]
    }

    def __init__(self,
                 cfg: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(UniqueMetic, self).__init__(component_config=cfg, **kwargs)

    # def prepare_data(self, **kwargs):
    #     pass

    def train(self,
              similes: List[Text, Mol],
              n_jobs=1,
              **kwargs) -> Tuple[Dict, Dict]:
        result = {}
        content = {}
        similes = remove_invalid(similes, canonize=True)
        for k in self.component_config["unique_k"]:
            result['unique@{}'.format(k)] = fraction_unique(gen=similes, k=k, n_jobs=n_jobs)

        return content, result

