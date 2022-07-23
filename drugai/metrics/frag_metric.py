#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:52
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : Frag_metric.py
from __future__ import annotations, print_function
from typing import Optional, Dict, Text, Any, List, Tuple
import logging
import os

from rdkit.Chem.rdchem import Mol
from moses.metrics import FragMetric as Frag, remove_invalid
from moses.utils import get_mol, mapper
from moses.script_utils import read_smiles_csv

from drugai.component import Component
from drugai.utils.io import read_smiles_zip

logger = logging.getLogger(__name__)


class FragMetric(Component):
    defaults = {
        "batch_size":512,
        "test_dir":None,
        "test_scaffolds_dir":None,
        "use_test": True,
        "use_test_scaffolds": True
    }
    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(FragMetric, self).__init__(component_config=component_config, **kwargs)

    def prepare_data(self,
                     filename: Text,
                     content: Dict,
                     **kwargs):

        if content.get("Frag/" + filename, None) is None:
            if content.get(filename, None) is None:
                if self.component_config[filename + "_dir"] is not None:
                    test_dir = self.component_config[filename + "_dir"]
                    test = read_smiles_csv(test_dir)
                else:
                    test_dir = os.path.join(os.path.dirname(__file__), "resources", filename + ".csv.gz")
                    test = read_smiles_zip(test_dir)
                content[filename] = test
            else:
                test = content[filename]
            mols = mapper(n_jobs=kwargs.get("n_jobs", 1))(get_mol, test)
            content['Frag/' + filename] = Frag(**kwargs).precalc(mols=mols)
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
        kwargs_frag = {'n_jobs': n_jobs, 'device': device, 'batch_size': self.component_config["batch_size"]}

        if content.get("Frag/test", None) is None and self.component_config["use_test"]:
            content = self.prepare_data(filename="test", content=content, **kwargs_frag)
            result['Frag/Test'] = Frag(**kwargs_frag)(gen=mols, pref=content['Frag/test'])
            logger.info("Frag/Test: %s" % (result['Frag/Test']))

        if content.get("Frag/test_scaffolds", None) is None and self.component_config["use_test_scaffolds"]:
            content = self.prepare_data(filename="test_scaffolds", content=content, **kwargs_frag)
            result['Frag/test_scaffolds'] = Frag(**kwargs_frag)(gen=mols, pref=content['Frag/test_scaffolds'])
            logger.info("Frag/test_scaffolds: %s" % (result['Frag/test_scaffolds']))

        return content, result