# -*- encoding: utf-8 -*-
'''
Filename         :molgan_preprocessor.py
Description      :
Time             :2022/08/10 23:10:49
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import annotations, print_function
import logging
from typing import Text, List, Any

from drugai.shared.preprocess.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class MolGANPreprocessor(Preprocessor):
    def ___init__(self, 
                  add_Hs:bool=False,
                  kekulize:bool=False,
                  filters:Any=None,
                  **kwargs):
        self.add_Hs = add_Hs
        self.kekulize = kekulize
        self.filters = filters
        super(MolGANPreprocessor, self).__init__(**kwargs)
    
    def pre_process(self, 
                    dataset:List
                    )->List:
        pass

    def get_max_length(self, dataset:List):
        if len(dataset)==0:
            logger.warning("number of dataset is 0")
            return 0

    def get_data_from_paths(self,
                            path: Text,
                            usecols: List = ["SMILES"],
                            **kwargs
                            ) -> List:
        dataset = self._load_data(path=path, usecols=usecols)
        # 前处理
        dataset = self.pre_process(dataset)
        

