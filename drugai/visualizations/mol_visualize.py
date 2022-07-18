#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/17 22:58
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : mol_visualize.py
from __future__ import annotations, print_function
from typing import Text, Tuple, List, Any, Optional

from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol


def single_image(mol: Mol,
                 save_file: Text = None,
                 size: Tuple[int, int] = (200, 200),
                 imageType: Text = "png",
                 legend: Text = None,
                 **kwargs) -> Any:
    img = Draw.MolToImage(mol, size=size, kekulize=True, fitImage=True, imageType=imageType, legend=legend)
    if save_file:
        img.save(save_file)
        return

    return img


def multiple_image(mols: List[Mol],
                   save_file: Text = None,
                   molsPerRow: int = 4,
                   subImgSize=(200, 200),
                   legends=[],
                   **kwargs) -> Any:
    img = Draw.MolsToGridImage(mols,
                               molsPerRow=molsPerRow,
                               subImgSize=subImgSize,
                               legends=legends)
    if save_file:
        img.save(save_file)
        return
    return img


def template_image(mols: Optional[List[Mol], Mol],
                   template: Mol,
                   save_file: Text = None,
                   molsPerRow: int = 4,
                   subImgSize=(200, 200),
                   legends=[],
                   **kwargs
                   ):
    AllChem.Compute2DCoords(template)
    if isinstance(mols, Mol):
        mols = [mols]
    for mol in mols:
        AllChem.GenerateDepictionMatching2DStructure(mol, template)  # 模板绘制法，固定公共子结构的取向
        mols.append(mol)

    multiple_image(mols=mols,
                   save_file=save_file,
                   molsPerRow=molsPerRow,
                   subImgSize=subImgSize,
                   legends=legends)
