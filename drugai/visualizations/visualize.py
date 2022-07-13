#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 19:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : visualizations.py
from __future__ import annotations, print_function
from typing import Optional, Text, List, Tuple
import logging

from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


class Visualize(object):
    def __init__(self):
        pass

    def show(self,
             smiles: Optional[Text, Mol, List],
             mode="single",  # "single", "multiple"
             **kwargs):
        if mode == "single":
            if isinstance(smiles, str):
                mol = Chem.MolFromSmiles(smiles)
                self.single_image(mol, **kwargs)
            elif isinstance(smiles, Mol):
                self.single_image(smiles, **kwargs)
            else:
                raise KeyError
        elif mode == "multiple":
            if isinstance(smiles, list):
                if isinstance(smiles[0], str):
                    mol = [Chem.MolFromSmiles(s) for s in smiles]
                    self.multiple_image(mols=mol, **kwargs)
                elif isinstance(smiles[0], Mol):
                    self.multiple_image(mols=smiles, **kwargs)
                else:
                    raise KeyError
            else:
                raise KeyError

    def single_image(self,
                     mol: Mol,
                     save_file: Text = None,
                     size: Tuple[int, int] = (200, 200),
                     imageType: Text = "png",
                     legend: Text = None
                     ):
        img = Draw.MolToImage(mol, size=size, kekulize=True, fitImage=True, imageType=imageType, legend=legend)
        if save_file:
            img.save(save_file)
            return

        return img

    def multiple_image(self,
                       mols: List[Mol],
                       save_file: Text = None,
                       molsPerRow: int = 4,
                       subImgSize=(200, 200),
                       legends=[],
                       **kwargs
                       ):
        img = Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=subImgSize, legends=legends)
        if save_file:
            img.save(save_file)
            return
        return img

    def template_image(self, mols):
        template = Chem.MolFromSmiles('c1nccc2n1ccc2')
        AllChem.Compute2DCoords(template)
        mols = []
        for smi in mols:
            mol = Chem.MolFromSmiles(smi)
            # 生成一个分子的描述，其中一部分 分子被约束为具有与参考相同的坐标。
            AllChem.GenerateDepictionMatching2DStructure(mol, template)
            mols.append(mol)

        # 基于分子文件输出分子结构
        img = Draw.MolsToGridImage(
            mols,  # mol对象
            molsPerRow=4,
            subImgSize=(200, 200),
            legends=['' for x in mols]
        )
        img.save('./mol12.jpg')

    def weight_image(self, mols):

        pass
        # 权重可视化 https://zhuanlan.zhihu.com/p/141440170?utm_source=wechat_session


