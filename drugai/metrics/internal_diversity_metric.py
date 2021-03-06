#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:04
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : internal_diversity_metric.py
from __future__ import annotations, print_function

import logging
from typing import Optional, Dict, Text, Any, List, Tuple

from moses.utils import get_mol, mapper
from moses.metrics import internal_diversity, remove_invalid
from rdkit.Chem.rdchem import Mol

from drugai.component import Component

logger = logging.getLogger(__name__)


class InternalDiversityMetric(Component):
    defaults = {
        "p": [1, 2]
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(InternalDiversityMetric, self).__init__(component_config=component_config, **kwargs)

    def train(self,
              smiles: List[Text, Mol],
              n_jobs=1,
              device="cpu",
              content: Dict = {},
              **kwargs: Any
              ) -> Tuple[Dict, Dict]:
        result = {}
        similes = remove_invalid(similes, canonize=True)  ## 移除无效的分子
        mols = mapper(n_jobs)(get_mol, similes)
        for p in self.component_config["p"]:
            result["IntDiv" + str(p)] = internal_diversity(gen=mols, n_jobs=n_jobs, device=device, p=p)
            logger.info("IntDiv%s: %s" %(p, result["IntDiv" + str(p)]))
        return content, result
