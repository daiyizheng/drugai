#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 19:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : visualizations.py
from __future__ import annotations, print_function

import os
import random
from typing import Optional, Text, List, Tuple, Any
import logging

from moses.script_utils import read_smiles_csv
from rdkit import Chem

from drugai.visualizations.mol_visualize import single_image, multiple_image, template_image

logger = logging.getLogger(__name__)


class Visualize(object):
    def __init__(self):
        pass

    def show(self,
             smiles1: Text,
             smiles2: Text = None,
             mode="single",
             # "single_image", "multiple_image", "template_image", "partial_charge_similarity_weights", "logp_similarity_weights", "fingerprint_similarity_weights"
             **kwargs):
        logger.info("mode is %s" % (mode))

        if mode == "single_image":
            mol = self.covert_data(smiles=smiles1)
            if isinstance(mol, list):
                mol = random.sample(mol, 1)[0]
            return single_image(mol=mol, **kwargs)

        elif mode == "multiple_image":
            mols = self.covert_data(smiles=smiles1)
            return multiple_image(mols=mols, **kwargs)

        elif mode == "template_image":
            template = kwargs.get("template", None)
            if template is None:
                logger.error("parameter `template` is missing")
                raise KeyError
            mol = self.covert_data(smiles=smiles1)
            if isinstance(mol, list):
                mol = random.sample(mol, 1)[0]
            return template_image(mols=mol, template=template, **kwargs)

        elif mode == "partial_charge_similarity_weights":
            pass
        elif mode == "logp_similarity_weights":
            pass
        elif  mode == "fingerprint_similarity_weights":
            pass
        else:
            raise KeyError

    def covert_data(self,
                    smiles:Text
                    ) -> None:
        if self.is_smiles(smiles=smiles):
            mol = self.smiles_to_mol(smiles=smiles)
        elif self.is_smiles_path(smiles=smiles):
            smiles = read_smiles_csv(path=smiles)
            if len(smiles) == 0:
                raise KeyError
            mol = [self.smiles_to_mol(s) for s in smiles if self.smiles_to_mol(s) is not None]
        else:
            logger.error("parameter `smiles` is not valid")
            raise KeyError
        if mol is None or (isinstance(mol, list) and len(mol)==0):
            raise KeyError

        return mol

    def is_smiles(self,
                  smiles: Text
                  ) -> bool:
        try:
            Chem.MolFromSmiles(smiles)
            return True
        except:
            return False

    def is_smiles_path(self,
                       smiles: Text
                       ) -> bool:
        if os.path.exists(smiles):
            return True
        return False

    def smiles_to_mol(self,
                      smiles: Text
                      ) -> Any:
        try:
            return Chem.MolFromSmiles(smiles)
        except:
            return None
