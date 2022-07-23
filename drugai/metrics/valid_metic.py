#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 11:23
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : valid_metic.py
from __future__ import annotations, print_function

import logging
from typing import Any, Optional, Dict, Text, List, Tuple

from rdkit.Chem.rdchem import Mol
from moses.metrics import fraction_valid

from drugai.component import Component

logger = logging.getLogger(__name__)


class ValidMetic(Component):
    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(ValidMetic, self).__init__(component_config=component_config, **kwargs)

    def train(self,
              smiles: List[Text, Mol],
              n_jobs=1,
              **kwargs) -> Tuple[Dict, Dict]:
        content = {}
        result = {"valid": fraction_valid(gen=smiles, n_jobs=n_jobs)}
        logger.info("valid: %s" %(result['valid']))
        return content, result
