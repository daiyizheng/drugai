#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:04
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : internal_diversity_metric.py
from __future__ import annotations, print_function

from typing import Optional, Dict, Text, Any, List, Tuple

from moses.utils import get_mol, mapper

from moses.metrics import internal_diversity, remove_invalid
from rdkit.Chem.rdchem import Mol

from drugai.component import Component


class InternalDiversityMetric(Component):
    defaults = {
        "p":[1,2]
    }

    def __init__(self,
                 cfg: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(InternalDiversityMetric, self).__init__(component_config=cfg, **kwargs)


    def train(self,
              similes: List[Text, Mol],
              n_jobs =1,
              device="cpu",
              content:Dict={},
              **kwargs: Any
              )->Tuple[Dict,Dict]:

        result = {}
        similes = remove_invalid(similes, canonize=True)  ## 移除无效的分子
        mols = mapper(n_jobs)(get_mol, similes)
        for p in self.component_config["p"]:
            result["IntDiv"+p] = internal_diversity(gen=mols,
                                                           n_jobs=n_jobs,
                                                           device=device,
                                                           p=p)
        return content, result