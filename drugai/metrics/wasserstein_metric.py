#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 13:57
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : logP_metric.py
from __future__ import annotations, print_function

import os, logging
from typing import Optional, Dict, Text, Any, List, Tuple

from drugai.utils.io import read_smiles_zip
from moses.utils import mapper, get_mol

from moses.script_utils import read_smiles_csv
from rdkit.Chem.rdchem import Mol

from moses.metrics import WassersteinMetric as wm, remove_invalid
from moses.metrics import logP, SA, QED, weight

from drugai.component import Component

logger = logging.getLogger(__name__)


class WassersteinMetric(Component):
    defaults = {
        "attributes": ["logP", "SA", "QED", "weight"],
        "test_dir": None
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(WassersteinMetric, self).__init__(component_config=component_config, **kwargs)
        self.attributes_funcs_map = {"logP": logP, "SA": SA, "QED": QED, "weight": weight}
        self.attributes_funcs = {k: self.attributes_funcs_map[k] for k in self.component_config["attributes"]}

    def prepare_data(self,
                     filename: Text,
                     type_name: Text,
                     content: Dict,
                     **kwargs) -> Dict:
        if content.get(type_name, None) is None:
            if content.get(filename, None) is None:
                if self.component_config[filename + "_dir"] is not None:
                    test_dir = self.component_config[filename + "_dir"]
                    test = read_smiles_csv(test_dir)
                else:
                    test_dir = os.path.join(os.path.dirname(__file__), "resources", filename + ".csv.gz")
                    test = read_smiles_zip(test_dir)
                logger.info("test path is %s" %(test_dir))
                content[filename] = test
            else:
                test = content[filename]
            mols = mapper(kwargs.get("n_jobs", 1))(get_mol, test)
            content[type_name] = wm(self.attributes_funcs_map[type_name], **kwargs).precalc(mols)
        return content

    def train(self,
              smiles: List[Text, Mol],
              n_jobs=1,
              device: Text = "cpu",
              content: Dict = {},
              **kwargs: Any
              ) -> Tuple[Dict, Dict]:
        similes = remove_invalid(smiles, canonize=True)  ## 移除无效的分子
        mols = mapper(n_jobs)(get_mol, similes)
        result = {}
        kwargs_wm = {"n_jobs": n_jobs}
        for name, func in self.attributes_funcs.items():
            logger.info("WassersteinMetric, function %s start" %(name))
            content = self.prepare_data(filename="test",
                                        content=content,
                                        type_name=name,
                                        **kwargs)
            result[name] = wm(func, **kwargs_wm)(gen=mols, pref=content[name])
            logger.info(name+": %s" %(result[name]))
            logger.info("WassersteinMetric, function %s end" % (name))

        return content, result
