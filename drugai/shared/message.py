#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/8/12 11:51 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : message.py
# @desc :
from __future__ import annotations, print_function
import logging
from typing import List, Optional, Dict, Text, Any, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

logger = logging.getLogger(__name__)

class Molecular:
    def __init__(self,
                 smiles:Text,
                 mol:Mol = None,
                 atom_nums:int = 0,
                 bond_nums:int = 0,
                 canonical_smiles:Text = None,
                 canonical_mol:Mol = None,
                 atom_matrix:np.ndarray=None,
                 adj_matrix:np.ndarray = None,
                 degree_matrix:np.ndarray = None,
                 laplace_matrix:np.ndarray = None,
                 feature_matrix:np.ndarray = None,
                 **kwargs):
        self.__smiles = smiles # 原子分子的smiles表示
        self.__mol = mol # 原子分子的rdkit mol对象
        self.__atom_nums = atom_nums # 原子分子的原子数
        self.__bond_nums = bond_nums # 原子分子的键数
        self.__canonical_smiles = canonical_smiles # 原子分子的规范化smiles表示
        self.__canonical_mol = canonical_mol # 原子分子的规范化rdkit mol对象
        self.__adj_matrix = adj_matrix # 原子分子的邻接矩阵
        self.__atom_matrix = atom_matrix
        self.__degree_matrix = degree_matrix # 原子分子的度矩阵
        self.__laplace_matrix = laplace_matrix # 原子分子的拉普拉斯矩阵
        self.__feature_matrix = feature_matrix # 原子分子的特征
    
    @property
    def smiles(self) -> Text:
        return self.__smiles
    
    @smiles.setter
    def smiles(self, value:Text):
        self.__smiles = value

    @property
    def mol(self) -> Mol:
        return self.__mol
    
    @mol.setter
    def mol(self, value: Mol):
        self.__mol = value
    
    @property
    def atom_nums(self) -> int:
        return self.__atom_nums
    
    @atom_nums.setter
    def atom_nums(self, value: int):
        self.__atom_nums = value
    
    @property
    def bond_nums(self) -> int:
        return self.__bond_nums

    @bond_nums.setter
    def bond_nums(self, value: int):
        self.__bond_nums = value
    
    @property
    def canonical_smiles(self):
        return self.__canonical_smiles
    
    @canonical_smiles.setter
    def canonical_smiles(self, value: Text):
        self.__canonical_smiles = value
    
    @property
    def canonical_mol(self) -> Mol:
        return self.__canonical_mol
    
    @canonical_mol.setter
    def canonical_mol(self, value: Mol):
        self.__canonical_mol = value
    
    @property
    def atom_matrix(self):
        return self.__atom_matrix

    @atom_matrix.setter
    def atom_matrix(self, value: np.ndarray):
        self.__atom_matrix = value
    
    @property
    def adj_matrix(self) -> np.ndarray:
        return self.__adj_matrix
    
    @adj_matrix.setter
    def adj_matrix(self, value: np.ndarray):
        self.__adj_matrix = value
    
    @property
    def degree_matrix(self) -> np.ndarray:
        return self.__degree_matrix
    
    @degree_matrix.setter
    def degree_matrix(self, value: np.ndarray):
        self.__degree_matrix = value
    
    @property
    def feature_matrix(self) -> np.ndarray:
        return self.__feature_matrix
    
    @feature_matrix.setter
    def feature_matrix(self, value: np.ndarray):
        self.__feature_matrix = value
    
    @property
    def laplace_matrix(self) -> np.ndarray:
        return self.__laplace_matrix
    
    @laplace_matrix.setter
    def laplace_matrix(self, value: np.ndarray):
        self.__laplace_matrix = value
    
    def __str__(self):
        return self.smiles
    
    def __len__(self):
        return len(self.smiles)

    def __repr__(self):
        return self.smiles      

class Message(Molecular):
    def __init__(
            self,
            smiles: Text = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(smiles=smiles, **kwargs)

    @staticmethod
    def get_atom_num(mol:Mol)->int:
        """
        Get atom number of molecule
        args:
            mol: molecule object
        return:number of atoms()
        """
        if mol is None:
            return 0
        return mol.GetNumAtoms()
    
    @staticmethod
    def get_bond_num(mol:Mol):
        return mol.GetNumBonds()

    @staticmethod
    def smiles_to_mol(smiles: Text):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            logger.error(f"{smiles} is not a valid smiles")
            mol = None
        return mol

    @staticmethod
    def mol_to_smiles(mol: Mol,
                      isomericSmiles: bool = False,
                      canonical: bool = True
                      ) -> Text:
        if mol is None:
            return None
        return Chem.MolToSmiles(mol,
                                isomericSmiles=isomericSmiles,
                                canonical=canonical)

    @staticmethod
    def add_Hs(mol: Mol) -> Mol:
        return Chem.AddHs(mol)

    @staticmethod
    def kekulize(mol: Mol) -> Mol:
        Chem.Kekulize(mol)

    @staticmethod
    def get_adj_ids(mol: Mol,
                    max_length: int,
                    bond_type_token_to_id: dict,
                    bond_padding_token=Chem.rdchem.BondType.ZERO,
                    connected=True
                    ) -> np.ndarray:
        A = np.zeros(shape=(max_length, max_length), dtype=np.int32)
        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [bond_type_token_to_id.get(b.GetBondType(), bond_type_token_to_id[bond_padding_token]) for b in
                     mol.GetBonds()]
        A[begin][end] = bond_type
        A[end][begin] = bond_type

        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)
        return A if connected and (degree > 0).all() else None

    @staticmethod
    def get_D(A: np.ndarray
              ) -> np.ndarray:
        return np.count_nonzero(A, axis=-1)
    
    @staticmethod
    def get_ori_atom_ids(mol: Mol, 
                         max_length:int=None
                         )->np.ndarray:
        ori_atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
        if max_length is None:
            return np.array(ori_atom_list, dtype=np.int32)

        n_atom = len(ori_atom_list)
        if max_length < len(ori_atom_list):
            raise ValueError(f"max_length {max_length} is smaller than the length of ori_atom_list {n_atom}")
    
        atom_array = np.zeros(max_length, dtype=np.int32)
        atom_array[:n_atom] = np.array(ori_atom_list, dtype=np.int32)
        return atom_array

    @staticmethod
    def get_atom_ids(mol: Mol,
                     max_length: int,
                     atom_token_to_id: dict,
                     atom_padding_token: int = 0
                     )->np.ndarray:
        atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
        if max_length is None:
            return np.array(atom_list, dtype=np.int32)
        
        n_atom = len(atom_list)
        if max_length < len(atom_list):
            raise ValueError(f"max_length {max_length} is smaller than the length of ori_atom_list {n_atom}")
    
        return np.array(
            [atom_token_to_id.get(atom, atom_token_to_id[atom_padding_token]) for atom in atom_list] + \
            [atom_token_to_id[atom_padding_token]] * (max_length - n_atom), dtype=np.int32)
    
    @staticmethod
    def get_smiles_ids(smiles: Text,
                       max_length: int,
                       smiles_token_to_ids:dict,
                       smiles_padding_token:Text="E"
                       )->np.ndarray:
        if isinstance(smiles, Mol):
            smiles = Message.mol_to_smiles(smiles)
            if smiles is None:
                return []
        return np.array([smiles_token_to_ids.get(c, smiles_token_to_ids[smiles_padding_token]) for c in smiles] \
            + [smiles_token_to_ids[smiles_padding_token]] * (max_length - len(smiles)), dtype=np.int32)

    ##   ----------------------------------  atom fearures  --------------------------------------------------------------
    def get_feature(self, 
                    mol: Mol,
                    d_num: int = 5,
                    explicit_valence_num: int = 9,
                    hybridization_num: list = [1, 7],
                    implicit_valence_num: int = 9,
                    explicit_hs_num: int = 5,
                    implicit_hs_num: int = 5,
                    radical_electrons_num: int = 5,
                    is_in_ring_size_num: list = [2, 9]
                    )->np.ndarray:
            if mol is None:
                return None
            d_by_num = self.get_D_by_nums(mol=mol, d_num=d_num)
            explicit_valence_by_num = self.get_explicit_valence(mol=mol, explicit_valence_num=explicit_valence_num)
            hybridization_by_num = self.get_hybridization(mol=mol, hybridization_num=hybridization_num)
            implicit_valence_by_num = self.get_implicit_valence(mol=mol, implicit_valence_num=implicit_valence_num)
            aromatic = self.get_is_aromatic()
            no_implicit = self.get_no_implicit()
            explicit_hs_by_num = self.get_num_explicit_hs(mol=mol, explicit_hs_num=explicit_hs_num)
            implicit_hs_by_num = self.get_num_implicit_hs(mol=mol, implicit_hs_num=implicit_hs_num)
            radical_electrons_by_num = self.get_num_radical_electrons(mol=mol, radical_electrons_num=radical_electrons_num)
            is_in_ring = self.get_is_in_ring()
            is_in_ring_size_by_num = self.get_is_in_ring_size(mol=mol, is_in_ring_size_num=is_in_ring_size_num)
            return np.concatenate([d_by_num, 
                                   explicit_valence_by_num, 
                                   hybridization_by_num, 
                                   implicit_valence_by_num,
                                   aromatic,
                                   no_implicit,
                                   explicit_hs_by_num, 
                                   implicit_hs_by_num, 
                                   radical_electrons_by_num, 
                                   is_in_ring,
                                   is_in_ring_size_by_num], axis=-1)


    @staticmethod
    def get_D_by_nums(mol:Mol,
                      num: int = 5
                      )->np.ndarray:
        if mol is None:
            return None
        return np.array([[atom.GetDegree() == i for i in range(num)] for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_explicit_valence(mol:Mol,
                             num: int = 9
                             )->np.ndarray:
        if mol is None:
            return None
        return np.array([[atom.GetExplicitValence() == i for i in range(num)]for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_hybridization(mol:Mol,
                          num: list = [1, 7]
                          )->np.ndarray:
        if mol is None:
            return None
        return np.array([[int(atom.GetHybridization()) == i for i in range(num[0], num[1])] \
            for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_implicit_valence(mol:Mol,
                             num: int = 9
                             )->np.ndarray:
        if mol is None:
            return None
        return np.array([[atom.GetImplicitValence() == i for i in range(num)] \
            for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_is_aromatic(mol:Mol)->np.ndarray:
        if mol is None:
            return None
        return np.array([[atom.GetIsAromatic()] for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_no_implicit(mol:Mol
                        )->np.ndarray:
        if mol is None:
            return None
        return np.array([[atom.GetNoImplicit()] for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_num_explicit_hs(mol:Mol,
                            num: int = 5
                            )->np.ndarray:
        if mol is None:
            return None
        return np.array([[atom.GetNumExplicitHs() == i for i in range(num)] \
            for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_num_implicit_hs(mol:Mol,
                            num: int = 5
                            )->np.ndarray:
        if mol is None:
            return None
        return np.array([[atom.GetNumImplicitHs() == i for i in range(num)] for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_num_radical_electrons(mol:Mol,
                                  num: int = 5
                                  )->np.ndarray:
        if mol is None:
            return None
        return np.array([[atom.GetNumRadicalElectrons() == i for i in range(num)] for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_is_in_ring(mol:Mol):
        if mol is None:
            return None
        return np.array([[atom.IsInRing()] for atom in mol.GetAtoms()], dtype=np.int32)

    @staticmethod
    def get_is_in_ring_size(mol:Mol, num:List=[2, 9]):
        if mol is None:
            return None
        return np.array([[atom.IsInRingSize(i) for i in range(num[0], num[1])] for atom in mol.GetAtoms()], dtype=np.int32)
    
    @staticmethod
    def get_le_lv(L:np.ndarray
                  )->Tuple[np.ndarray, np.ndarray]:
        if L is None:
            return None, None
        Le, Lv = np.linalg.eigh(L)
        return Le, Lv
    
    # ----------------------------------  atom features  --------------------------------------------------------------
