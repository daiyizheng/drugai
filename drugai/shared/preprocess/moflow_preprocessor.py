#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/8/4 7:44 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : moflow_preprocessor.py
# @desc :
from __future__ import annotations, print_function
import logging
import traceback
from typing import List, Text, Tuple, Dict

import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from drugai.models.dataset import NumpyTupleDataset
from ..message import Message
from drugai.shared.preprocess.preprocessor import Preprocessor
from drugai.utils.common import type_check_num_atoms

logger = logging.getLogger(__name__)


class MoFlowPreprocessor(Preprocessor):
    def __init__(self, 
                 max_atoms:int=None,
                 **kwargs
                 )->None:
        if max_atoms is None:
            raise ValueError('max_atoms must be provided')
        self.max_atoms = max_atoms
        self.bond_type_token_to_id = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }
        super(MoFlowPreprocessor, self).__init__(**kwargs)
    
    def prepare_smiles_and_mol(self, mol):
        """Prepare `smiles` and `mol` used in following preprocessing.
        This method is called before `get_input_features` is called, by parser
        class.
        This method may be overriden to support custom `smile`/`mol` extraction
        """
        # Note that smiles expression is not unique.
        # we obtain canonical smiles which is unique in `mol`
        canonical_smiles = Message.mol_to_smiles(mol=mol, isomericSmiles=False, canonical=True)
        canonical_mol = Message.smiles_to_mol(canonical_smiles)
        if self.add_Hs:
            canonical_mol = Message.add_Hs(canonical_mol)
        if self.kekulize:
            Message.kekulize(canonical_mol)
        return canonical_smiles, canonical_mol


    def get_input_features(self, 
                           mol:Mol
                           )->Tuple[np.ndarray, np.ndarray]:

        type_check_num_atoms(mol, self.max_atoms)
        atom_array = Message.get_ori_atom_ids(mol=mol, max_length=self.max_atoms)
        adj_array = self.construct_discrete_edge_matrix(mol, out_size=self.max_atoms)
        return atom_array, adj_array

    def construct_discrete_edge_matrix(self, 
                                       mol:Mol, 
                                       out_size:int=None
                                       )->np.ndarray:
        """Returns the edge-type dependent adjacency matrix of the given molecule."""
        if mol is None:
            raise KeyError('mol is None')
        
        N = Message.get_atom_num(mol=mol)

        if out_size is None or out_size <=0:
            raise ValueError("out_size must be provided and > 0")
        elif out_size >= N:
            size = out_size
        else:
            raise ValueError(
                'out_size {} is smaller than number of atoms in mol {}, smiles is {}'
                .format(out_size, N, Message.mol_to_smiles(mol=mol)))

        adjs = np.zeros((4, size, size), dtype=np.float32)

        
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            ch = self.bond_type_token_to_id[bond_type]
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adjs[ch, i, j] = 1.0
            adjs[ch, j, i] = 1.0
        return adjs     
        
    def pre_process(self, 
                    dataset:List,
                    **kwargs
                    )->Tuple:
        """Preprocess the dataset.
        This method is called before `get_input_features` is called, by parser
        """
        ## filter mol             
        logger.info("Preprocessing dataset start...")
        c_mols = []
        c_smiles = []
        for message in tqdm(dataset, total=len(dataset)):
            try:
                mol = Message.smiles_to_mol(message.smiles)
                canonical_smiles, mol = self.prepare_smiles_and_mol(mol)
                c_mols.append(mol)
                c_smiles.append(canonical_smiles)
                
            except Exception as e:
                logger.warning('parse(), type: {}, {}' .format(type(e).__name__, e.args))
                logger.info(traceback.format_exc())
                continue
        
        if self.__dict__.get("max_atoms", None) is not None:
             c_mols = self.filter_mol(c_mols)
        
        if self.__dict__.get('bond_type_token_to_id', None) is None:
            self.bond_type_token_to_id, self.bond_type_id_to_token = self.bond_type_map(c_mols)
        
        logger.info("load atom features and adj features start...")
        messages = []
        for c_m, c_s in tqdm(list(zip(c_mols,c_smiles))):
            atom_features, adj_features = self.get_input_features(c_m)
            messages.append(Message(smiles=c_s, atom_matrix=atom_features, adj_matrix=adj_features))
        logger.info("load atom features and adj features end...")

        logger.info("Preprocessing dataset end...")
        return NumpyTupleDataset(datasets = messages, **kwargs)