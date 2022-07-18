#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/17 22:58
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : atom_visualize.py
from __future__ import annotations, print_function

from typing import Text, Tuple, Any

from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import rdMolDescriptors


def partial_charge_similarity_weights(mol: Mol,
                                      save_file: Text = None,
                                      size: Tuple[int, int] = (300, 300),
                                      scale: int = 10,
                                      contourLines: int = 10,
                                      **kwargs
                                      ) -> Any:
    """
    原子partial charge可视化

    """
    AllChem.ComputeGasteigerCharges(mol)
    charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,
                                                     charges,
                                                     size=size,
                                                     contourLines=contourLines,
                                                     scale=scale)
    if save_file:
        fig.savefig(save_file, bbox_inches='tight')
        return
    return fig


def logp_similarity_weights(mol: Mol,
                            save_file: Text = None,
                            size: Tuple[int, int] = (300, 300),
                            scale: int = 10,
                            contourLines: int = 10,
                            **kwargs
                            ) -> Any:
    """
    原子logP可视化
    """
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,
                                                     [x for x, y in contribs],
                                                     size=size,
                                                     contourLines=contourLines,
                                                     scale=scale)
    if save_file:
        fig.savefig(save_file, bbox_inches='tight')
        return
    return fig


def fingerprint_similarity_weights(mol1:Mol,
                                   mol2:Mol,
                                   save_file: Text = None,
                                   size: Tuple[int, int] = (300, 300),
                                   scale: int = 10,
                                   contourLines: int = 10,
                                   **kwargs
                                   ) -> Any:
    weights = SimilarityMaps.GetAtomicWeightsForFingerprint(mol1, mol2, SimilarityMaps.GetMorganFingerprint)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol2, weights, size=size, scale=scale, contourLines=contourLines)
    if save_file:
        fig.savefig(save_file, bbox_inches='tight')
        return
    return fig

