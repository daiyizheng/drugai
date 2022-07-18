#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 19:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : visualizations.py
from __future__ import annotations, print_function

import os
from typing import Optional, Text, List, Tuple
import logging

from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem

from drugai.visualizations.mol_visualize import single_image

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
        if mode == "single_image":
            if not self.is_smiles(smiles=smiles1):
                raise KeyError
            mol = self.smiles_to_mol(smiles=smiles1)
            return single_image(mol=mol, **kwargs)

    def is_smiles(self, smiles: Text) -> bool:
        try:
            Chem.MolFromSmiles(smiles)
            return True
        except Exception as e:
            return False

    def is_smiles_path(self,
                       smiles: Text
                       ) -> bool:
        if os.path.exists(smiles):
            return True
        return False

    def smiles_to_mol(self, smiles:Text):
        return Chem.MolFromSmiles(smiles)

