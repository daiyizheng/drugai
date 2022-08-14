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
from typing import Text, Any, List, Union

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..message import Message
from drugai.shared.preprocess.preprocessor import Preprocessor
from drugai.models.generate.gen_vocab import Vocab

logger = logging.getLogger(__name__)


class BasicPreprocessor(Preprocessor):
    def __init__(self,  **kwargs):
        super(BasicPreprocessor, self).__init__(**kwargs)
    
    def pre_process(self, 
                    dataset:List
                    )->List:
        logger.info("Pre-processing data start")

        logger.info("Pre-processing data end")
        return dataset

  
    def build_vocab(self, 
                    vocab: Vocab, 
                    data: Union[List[Message], List[Text], List[np.ndarray]])->"Vocab":
        logger.info("Building vocab start")
        if not isinstance(data, list):
            raise KeyError("The data only support list")
        if isinstance(data[0], Message):
            data = [d.smiles for d in tqdm(data)]
        elif isinstance(data[0], Text) or isinstance(data[[0], np.ndarray]):
            data = data
        else:
            raise KeyError("The data only support list[Message] or list[Text] or list[np.ndarray]")
        logger.info("Building vocab end")
        return vocab.from_data(data)
