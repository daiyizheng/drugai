#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:52
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : Scaf_metric.py
from __future__ import annotations, print_function

import logging
import os
from typing import Optional, Dict, Text, Any, Tuple, List

from rdkit.Chem.rdchem import Mol
from moses.metrics import ScafMetric as Scaf, remove_invalid
from moses.utils import mapper, get_mol
from moses.script_utils import read_smiles_csv

from drugai.utils.io import read_smiles_zip
from drugai.component import Component

logger = logging.getLogger(__name__)


class ScafMetric(Component):
    defaults = {
        "batch_size": 512,
        "test_dir": None,
        "test_scaffolds_dir": None,
        "use_test": True,
        "use_test_scaffolds": True
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs: Any):
        super(ScafMetric, self).__init__(component_config=component_config, **kwargs)

    def prepare_data(self,
                     filename: Text,
                     content: Dict,
                     **kwargs):

        if content.get("Scaf/" + filename, None) is None:
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
            content['Scaf/' + filename] = Scaf(**kwargs).precalc(mols=mols)
        return content

    def train(self,
              smiles: List[Text, Mol],
              device: Text = "cpu",
              n_jobs: int = 1,
              content: Dict = {},
              **kwargs: Any
              ) -> Tuple[Dict, Dict]:
        result = {}
        gen = remove_invalid(smiles, canonize=True)
        mols = mapper(n_jobs=n_jobs)(get_mol, gen)
        kwargs_scaf = {'n_jobs': n_jobs, 'device': device, 'batch_size': self.component_config["batch_size"]}

        if content.get("Scaf/test", None) is None and self.component_config["use_test"]:
            content = self.prepare_data(filename="test", content=content, **kwargs_scaf)
            result['Scaf/Test'] = Scaf(**kwargs_scaf)(gen=mols, pref=content['Scaf/test'])
            logger.info("Scaf/Test: %s" % (result['Scaf/Test']))

        if content.get("Scaf/test_scaffolds", None) is None and self.component_config["use_test_scaffolds"]:
            content = self.prepare_data(filename="test_scaffolds", content=content, **kwargs_scaf)
            result['Scaf/test_scaffolds'] = Scaf(**kwargs_scaf)(gen=mols, pref=content['Scaf/test_scaffolds'])
            logger.info("Scaf/test_scaffolds: %s" % (result['Scaf/test_scaffolds']))

        return content, result
