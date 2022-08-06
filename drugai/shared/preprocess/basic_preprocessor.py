#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/8/4 10:48 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : basic_preprocessor.py
# @desc :
from __future__ import annotations, print_function
import logging
import os
from typing import Text, Any, List

from tqdm import tqdm

from drugai.shared.preprocess.preprocessor import Preprocessor
from drugai.utils.io import read_smiles_csv

logger = logging.getLogger(__name__)


class BasicPreprocessor(Preprocessor):
    def __init__(self, 
                 transform=None,
                 **kwargs):
        self.transform = transform
        super(BasicPreprocessor, self).__init__(**kwargs)

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
