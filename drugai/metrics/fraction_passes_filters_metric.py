#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:20
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : fraction_passes_filters_metric.py
from __future__ import annotations, print_function

import logging
from typing import Optional, Dict, Text, Any, List, Tuple

from moses.utils import get_mol, mapper
from moses.metrics import remove_invalid, fraction_passes_filters
from rdkit.Chem.rdchem import Mol

from drugai.component import Component

logger = logging.getLogger(__name__)


class FractionPassesFiltersMetric(Component):
    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(FractionPassesFiltersMetric, self).__init__(component_config=component_config, **kwargs)

    def train(self,
              smiles: List[Text, Mol],
              n_jobs=1,
              device: Text = "cpu",
              content: Dict = {},
              **kwargs: Any
              ) -> Tuple[Dict, Dict]:
        similes = remove_invalid(smiles, canonize=True)  ## 移除无效的分子
        mols = mapper(n_jobs)(get_mol, similes)
        result = {'Filters': fraction_passes_filters(mols, n_jobs)}
        logger.info("Filters: %s" % (result["Filters"]))

        return content, result
