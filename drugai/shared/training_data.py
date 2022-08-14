#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 13:21
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : training_data.py
from __future__ import annotations, print_function

import logging
from typing import Dict, Text, Any, List

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from drugai.models.generate.gen_vocab import Vocab
from drugai.shared.preprocess.preprocessor import Preprocessor
from .message import Message

logger = logging.getLogger(__name__)

    
class TrainingData:
    def __init__(self,
                 train_data: List[Message],
                 eval_data: List[Message]=None,
                 test_data: List[Message]=None,
                 num_workers: int = 0,
                 preprocessor:Preprocessor=None):
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.num_workers = num_workers
        self.preprocessor = preprocessor


    def dataloader(self,
                   batch_size: int,
                   collate_fn: Any,
                   shuffle: bool = True,
                   mode: Text = "train"):

        if mode == "train":
            dataset = self.train_data
        elif mode == "eval":
            dataset = self.eval_data
        elif mode == "test":
            dataset = self.test_data
        else:
            raise KeyError
        
        dataset = self.preprocessor.pre_process(dataset=dataset)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn)
    
    @property
    def get_train_data(self):
        return self.train_data
    
    @property
    def get_eval_data(self):
        return self.eval_data
    
    @property
    def get_test_data(self):
        return self.test_data

    @property
    def get_atom_list(self):
        atom_token_to_id = getattr(self.preprocessor, "atom_token_to_id", None)
        if atom_token_to_id is not None:
            return atom_token_to_id.keys()
        return self.preprocessor.atom_list(self.train_data)
    
    @property
    def get_atom_token_to_id(self):
        atom_token_to_id = getattr(self.preprocessor, "atom_token_to_id", None)
        if atom_token_to_id is not None:
            return atom_token_to_id
        return {v:i for i, v in enumerate(self.get_atom_list)}
    
    @property
    def get_bond_list(self):
        bond_type_token_to_id = getattr(self.preprocessor, "bond_type_token_to_id", None)
        if bond_type_token_to_id is None:
            raise ValueError("bond_type_token_to_id is None")
        bond_list = [t[0] for t in sorted(bond_type_token_to_id.items(), key=lambda x: x[1])]
        return bond_list
    
    @property
    def get_bond_token_to_id(self):
        bond_type_token_to_id = getattr(self.preprocessor, "bond_type_token_to_id", None)
        if bond_type_token_to_id is None:
            raise ValueError("bond_type_token_to_id is None")
        return bond_type_token_to_id
    
    def get_vocab(self,
                vocab: Vocab
                ) -> "Vocab":
        return self.preprocessor.build_vocab(vocab=vocab, data=self.train_data)

            
    
