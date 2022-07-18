#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:52
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : SNN_metric.py
from __future__ import annotations, print_function

import os
from typing import Optional, Dict, Text, Any, List, Tuple

from drugai.utils.io import read_smiles_zip
from moses.script_utils import read_smiles_csv

from moses.utils import mapper, get_mol
from rdkit.Chem.rdchem import Mol
from moses.metrics import remove_invalid
from moses.metrics import SNNMetric as SNN

from drugai.component import Component


class SNNMetric(Component):
    defaults = {
        "batch_size": 512,
        "test_dir": None,
        "test_scaffolds_dir": None,
        "use_test": True,
        "use_test_scaffolds": True
    }

    def __init__(self,
                 cfg: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(SNNMetric, self).__init__(component_config=cfg, **kwargs)

    def prepare_data(self,
                     filename:Text,
                     content:Dict,
                     **kwargs):

        if content.get("SNN/" + filename, None) is None:
            if content.get(filename, None) is None:
                if self.component_config[filename + "_dir"] is None:
                    test_dir = self.component_config[filename + "_dir"]
                    test = read_smiles_csv(test_dir)
                else:
                    test_dir = os.path.join(os.path.dirname(__file__), "resources", filename + ".csv.gz")
                    test = read_smiles_zip(test_dir)
                content[filename] = test
            else:
                test = content[filename]
            mols = mapper(n_jobs=kwargs.get("n_jobs", 1))(get_mol, test)
            content['SNN/' + filename] = SNN(**kwargs).precalc(mols=mols)
        return content

    def train(self,
              smiles:List[Text, Mol],
              device: Text = "cpu",
              n_jobs:int=1,
              content:Dict={},
              **kwargs: Any
              ) -> Tuple[Dict, Dict]:
        result = {}
        gen = remove_invalid(smiles, canonize=True)
        mols = mapper(n_jobs=n_jobs)(get_mol, gen)

        kwargs_snn = {'n_jobs': n_jobs, 'device': device, 'batch_size': self.component_config["batch_size"]}

        if content.get("SNN/test", None) is None and self.component_config["use_test"]:
            content = self.prepare_data(filename="test", content=content, **kwargs_snn)
            result['SNN/Test'] = SNN(**kwargs_snn)(gen=mols, pref=content['SNN/test'])

        if content.get("SNN/test_scaffolds", None) is None and self.component_config["use_test_scaffolds"]:
            content = self.prepare_data(filename="test_scaffolds", content=content,  **kwargs_snn)
            result['SNN/test_scaffolds'] = SNN(**kwargs_snn)(gen=mols, pref=content['SNN/test_scaffolds'])

        return content, result
