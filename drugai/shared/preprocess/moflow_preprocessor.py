#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/8/4 7:44 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : moflow_preprocessor.py
# @desc :
from __future__ import annotations, print_function
import logging
from typing import List, Text

from drugai.shared.preprocess.preprocessor import Preprocessor
from drugai.utils.io import read_smiles_csv

logger = logging.getLogger(__name__)


class MoFlowPreprocessor(Preprocessor):
    def __init__(self, 
                 max_atoms:int,
                 add_Hs:bool=False,
                 kekulize:bool=False,
                 transform=None,
                 **kwargs
                 ):
        self.transform = transform
        super(MoFlowPreprocessor, self).__init__(**kwargs)
    
    def _load_data(self, 
                   path:Text, 
                   usecols: List = ["SMILES"],
                   ):
        return read_smiles_csv(path=path, usecols=usecols)


    def pre_process(self, 
                    dataset:List
                    )->List:
        
        return dataset

    def get_data_from_paths(self,
                            path: Text,
                            usecols: List = ["SMILES"],
                            **kwargs
                            ) -> List:
        if path is None:
            dataset = []
        else:
            dataset = self._load_data(path=path, usecols=usecols)
        # 前处理
        dataset = self.pre_process(dataset)
        
        if self.transform is not None:
            dataset = [self.transform (d)for d in dataset]

        return dataset