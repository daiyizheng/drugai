#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:52
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : FCD_metric.py
from __future__ import annotations, print_function

import os
from typing import Optional, Dict, Text, Any, List, Tuple

from fcd_torch import FCD
from moses.utils import get_mol, mapper

from moses.metrics import remove_invalid

from moses.script_utils import read_smiles_csv, read_smiles_zip
from rdkit.Chem.rdchem import Mol

from drugai.component import Component


class FCDMetric(Component):
    defaults = {
        "batch_size":512,
        "test_dir":None,
        "test_scaffolds_dir":None,
        "use_test": True,
        "use_test_scaffolds": True
    }
    def __init__(self,
                 cfg: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(FCDMetric, self).__init__(component_config=cfg, **kwargs)


    def prepare_data(self,
                     filename:Text,
                     content:Dict,
                     **kwargs):
        if content.get("FCD/"+filename, None) is  None:
            if content.get(filename, None) is None:
                if self.component_config[filename+"_dir"] is None:
                    test_dir = self.component_config[filename+"_dir"]
                    test = read_smiles_csv(test_dir)
                else:
                    test_dir = os.path.join(os.path.dirname(__file__), "resources", filename+".csv.gz")
                    test = read_smiles_zip(test_dir)
                content[filename] = test
            else:
                test = content[filename]

            content['FCD/'+filename] = FCD(**kwargs).precalc(smiles_list=test)
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

        kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': self.component_config["batch_size"]}
        if content.get("FCD/test", None) is None and self.component_config["use_test"]:
            content = self.prepare_data(filename="test",content=content,  **kwargs_fcd)
            result['FCD/Test'] = FCD(**kwargs_fcd)(gen=gen, pref=content['FCD/test'])

        if content.get("FCD/test_scaffolds", None) is None and self.component_config["use_test_scaffolds"]:
            content = self.prepare_data(filename="test_scaffolds", content=content, **kwargs_fcd)
            result['FCD/test_scaffolds'] = FCD(**kwargs_fcd)(gen=gen, pref=content['FCD/test_scaffolds'])

        return content, result