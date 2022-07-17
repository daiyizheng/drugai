#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:20
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : fraction_passes_filters_metric.py
from __future__ import annotations, print_function

from typing import Optional, Dict, Text, Any, List, Tuple

from moses.utils import get_mol, mapper

from moses.metrics import remove_invalid, fraction_passes_filters
from rdkit.Chem.rdchem import Mol

from drugai.component import Component


class FractionPassesFiltersMetric(Component):
    def __init__(self,
                 cfg: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(FractionPassesFiltersMetric, self).__init__(component_config=cfg, **kwargs)


    def train(self,
              similes: List[Text, Mol],
              n_jobs=1,
              device:Text="cpu",
              content:Dict={},
              **kwargs: Any
              ) -> Tuple[Dict, Dict]:
        similes = remove_invalid(similes, canonize=True)  ## 移除无效的分子
        mols = mapper(n_jobs)(get_mol, similes)
        return content, {'Filters': fraction_passes_filters(mols, n_jobs)}