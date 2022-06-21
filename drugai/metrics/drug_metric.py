#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 0:07
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : drug_metric.py
from __future__ import annotations, print_function

from typing import Union, List

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

"""
评价指标
esr(effective structure rate): 分子有效结构率
logP：油水分配系数(LogP)是反映化合物水(脂)溶性大小的重要参数,水溶性差、生物利用度不高 Wildman and G. M. Crippen JCICS 39 868-873 (1999)。
QED：是一种将药物相似性量化为介于0和1之间的数值的方法。
weight: 分子质量
TPSA: 分子拓扑极表面积
"""

class DrugMetric(object):
    def __init__(self):
        pass

    @staticmethod
    def logP(mol: Mol):
        return Chem.Crippen.MolLogP(mol)

    @staticmethod
    def QED(mol: Mol):
        return Chem.QED.qed(mol)

    @staticmethod
    def effective_structure_rate(smiles: List[Mol]):
        return 1 - smiles.count(None) / len(smiles)

    @staticmethod
    def weight(mol: Mol):
        return Chem.Description.MolWt(mol)



    @staticmethod
    def covert_smiles_to_mol(smiles: Union[str, Mol]):
        if len(smiles) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol

    @staticmethod
    def covert_mol_to_smiles(mol:Mol):
        return Chem.MolToSmiles(mol)
