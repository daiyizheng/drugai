#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 13:20
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : unique_metric.py
from __future__ import annotations, print_function

import logging
from typing import Optional, Dict, Text, Any, List, Tuple

from moses.metrics import fraction_unique, remove_invalid
from rdkit.Chem.rdchem import Mol

from drugai.component import Component

logger = logging.getLogger(__name__)


class UniqueMetic(Component):
    defaults = {
        "unique_k": [1000, 10000]
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(UniqueMetic, self).__init__(component_config=component_config, **kwargs)

    def train(self,
              smiles: List[Text, Mol],
              n_jobs=1,
              **kwargs) -> Tuple[Dict, Dict]:
        result = {}
        content = {}
        similes = remove_invalid(smiles, canonize=True)
        for k in self.component_config["unique_k"]:
            result['unique@{}'.format(k)] = fraction_unique(gen=similes, k=k, n_jobs=n_jobs)
            logger.info("unique@%s: %s" % (k, result['unique@{}'.format(k)]))
        return content, result
