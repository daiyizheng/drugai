#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/8/4 7:45 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : preprocessor.py
# @desc :
from __future__ import annotations, print_function
import logging
import abc
from typing import List, Optional, Text, Any, Union

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from ..message import Message

from drugai.utils.io import read_smiles_csv

logger = logging.getLogger(__name__)


class Preprocessor(abc.ABC):
    @abc.abstractmethod
    def __init__(self,
                 add_Hs:bool=False,
                 kekulize:bool=False,
                 **kwargs
                 ) -> None:
        self.add_Hs = add_Hs
        self.kekulize = kekulize
        self.d_num = kwargs.get("d_num", 5)
        self.explicit_valence_num = kwargs.get("explicit_valence_num", 9)
        self.hybridization_num = kwargs.get("hybridization_num", [1,7])
        self.implicit_valence_num = kwargs.get("implicit_valence_num", 9)
        self.explicit_hs_num = kwargs.get("explicit_hs_num", 5)
        self.implicit_hs_num = kwargs.get("implicit_hs_num", 5)
        self.radical_electrons_num = kwargs.get("radical_electrons_num", 5)
        self.is_in_ring_size_num = kwargs.get("is_in_ring_size_num", [2, 9])

    
    @abc.abstractmethod
    def pre_process(self, data:List[Any], **kwargs)->Any:
        ...

    @abc.abstractmethod
    def get_data_from_paths(self, path:Text, **kwargs) -> Any:
        ...
    
    def _load_data(self, path:Text, usecols: List = ["SMILES"]) -> pd.Series:
        return read_smiles_csv(path=path, usecols=usecols)
    
    @staticmethod
    def get_atom_max_length(mols:List[Mol])->int:
        if len(mols) and isinstance(mols[0], str):
            mols = [Chem.MolFromSmiles(m) for m in mols]
        return max(mol.GetNumAtoms() for mol in mols)
    
    @staticmethod
    def get_smiles_max_length(smiles:List[Text])->int:
        if len(smiles) and isinstance(smiles[0], Mol):
            smiles = [Chem.MolToSmiles(mol) for mol in smiles]
        return max(len(smile) for smile in smiles)
    
    @staticmethod
    def filter_mol(mols:Union[List[Mol], List[Message]], 
                   nums:int=9)->List[Mol]:
        if not mols:
            return []
        if isinstance(mols[0], Message):
            mols = [Message.smiles_to_mol(m.smiles) for m in mols]
        return list(filter(lambda x: x.GetNumAtoms() <= nums, mols))

    @staticmethod
    def atom_list(mols:Union[List[Mol], List[Message]], 
                  atom_padding_token:int=0
                 )->List[int]:
        if not mols:
            return []
        if isinstance(mols[0], Message):
            mols = [Message.smiles_to_mol(m.smiles) for m in mols]
        atom_labels = list(set([atom.GetAtomicNum() for mol in mols for atom in mol.GetAtoms()]))
        if atom_padding_token in atom_labels:
            raise ValueError("atom_padding_token is in atom_labels")
        atom_labels = sorted(atom_labels + [atom_padding_token])
        return atom_labels

    @staticmethod
    def atom_type_map(mols:List[Mol], 
                      atom_padding_token:int=0):
        atom_labels = Preprocessor.atom_list(mols, atom_padding_token)
        atom_token_to_id = {l: i for i, l in enumerate(atom_labels)}
        atom_id_to_token = {i: l for i, l in enumerate(atom_labels)}
        return atom_token_to_id, atom_id_to_token
    
    @staticmethod
    def bond_list(mols:Union[List[Mol],List[Message]], 
                  bond_padding_token=Chem.rdchem.BondType.ZERO):
        if not mols:
            return []
        if isinstance(mols[0], Message):
            mols = [Message.smiles_to_mol(m.smiles) for m in mols]
        bond_labels = list(sorted(set(bond.GetBondType() for mol in mols for bond in mol.GetBonds())))
        if bond_padding_token in bond_labels:
            raise ValueError("bond_padding_token is in bond_labels")
        bond_labels = [bond_padding_token] + bond_labels
        return bond_labels
    
    @staticmethod
    def bond_type_map(mols:Union[List[Mol],List[Message]], 
                      bond_padding_token=Chem.rdchem.BondType.ZERO):
        bond_labels = Preprocessor.bond_list(mols, bond_padding_token)
        bond_type_token_to_id = {l: i for i, l in enumerate(bond_labels)}
        bond_type_id_to_token = {i: l for i, l in enumerate(bond_labels)}
        return bond_type_token_to_id, bond_type_id_to_token

    @staticmethod
    def smiles_list(mols:List, 
                    smiles_padding_token:Text='E'):
        if not mols:
            return []
        if isinstance(mols[0], Message):
            mols = [Message.smiles_to_mol(m.smiles) for m in mols]
        smiles_labels = list(set(c for mol in mols for c in Chem.MolToSmiles(mol)))
        if smiles_padding_token in smiles_labels:
            raise ValueError("smiles_padding_token is in smiles_labels")
        smiles_labels = [smiles_padding_token] + smiles_labels
        return smiles_labels
    
    @staticmethod
    def smiles_type_map(mols:List[Mol],
                        smiles_padding_token='E'):
        smiles_labels = Preprocessor.smiles_list(mols, smiles_padding_token)
        smiles_type_token_to_id = {l: i for i, l in enumerate(smiles_labels)}
        smiles_type_id_to_token = {i: l for i, l in enumerate(smiles_labels)}
        return smiles_type_token_to_id, smiles_type_id_to_token
    
    def get_data_from_paths(self,
                        path: Text,
                        usecols: List = ["SMILES"],
                        **kwargs
                        ) -> List:
        logger.info("Loading data from {} start".format(path))
        if path is None:
            dataset = pd.Series([])
        else:
            dataset = self._load_data(path=path, usecols=usecols)
        # 转为List[Message]
        messages = []
        if isinstance(dataset, pd.Series): 
            for m in tqdm(dataset):
                messages.append(Message(smiles=m, **kwargs))
        else:
            raise ImportError("The dataset only support pandas.Series")
        logger.info("Loading data from {} end".format(path))
        return messages

    


        

        




    

    

