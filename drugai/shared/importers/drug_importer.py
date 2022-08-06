#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 11:01
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : drug_importer.py
from __future__ import annotations, print_function
import os
import logging
from typing import Optional, Text, Union, List, Dict

import drugai.utils.io
from drugai.shared.importers.training_data_importer import TrainingDataImporter
from drugai.shared.preprocess.preprocessor import Preprocessor
from drugai.shared.training_data import TrainingData

logger = logging.getLogger(__name__)


class DrugImporter(TrainingDataImporter):
    def __init__(self,
                 config_file: Optional[Text] = None,
                 train_data_paths: Optional[Union[List[Text], Text]] = None,
                 eval_data_paths: Optional[Union[List[Text], Text]] = None,
                 test_data_paths: Optional[Union[List[Text], Text]] = None,
                 **kwargs) -> None:
        super(DrugImporter, self).__init__(config_file=config_file,
                                           train_data_paths=train_data_paths,
                                           eval_data_paths=eval_data_paths,
                                           test_data_paths=test_data_paths)

    def get_config(self) -> Dict:
        if not self.config_file or not os.path.exists(self.config_file):
            logger.debug("No configuration file was provided to the RasaFileImporter.")
            return {}
        return drugai.utils.io.read_config_yaml(self.config_file)

    def get_data(self,
                 preprocessor:Preprocessor,
                 **kwargs
                 ) -> "TrainingData":
        train_data = preprocessor.get_data_from_paths(path=self.train_data_paths)
        eval_data = preprocessor.get_data_from_paths(path=self.eval_data_paths)
        test_data = preprocessor.get_data_from_paths(path=self.test_data_paths)
        return TrainingData(train_data=train_data,
                            eval_data=eval_data,
                            test_data=test_data,
                            **kwargs)

    
