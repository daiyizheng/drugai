#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 11:23
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : valid_metic.py
from __future__ import annotations, print_function

from typing import Any, Optional, Dict, Text, List

from rdkit.Chem.rdchem import Mol
from moses.metrics import fraction_valid

from drugai.component import Component


class ValidMetic(Component):
    def __init__(self,
                 cfg: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(ValidMetic, self).__init__(component_config=cfg, **kwargs)

    def train(self, similes: List[Text, Mol], n_jobs=1, **kwargs):
        return {"valid": fraction_valid(gen=similes, n_jobs=n_jobs)}
