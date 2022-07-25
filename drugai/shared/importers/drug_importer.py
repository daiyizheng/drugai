#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 11:01
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : drug_importer.py
from __future__ import annotations, print_function
from typing import Optional, Text, Union, List, Dict
import os
import logging

import drugai.utils.io
from drugai.shared.importers.training_data_importer import TrainingDataImporter
from drugai.shared.loading import training_data_from_paths
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
                 mode:Text,
                 **kwargs
                 ) -> "TrainingData":
        train_data = training_data_from_paths(self.train_data_paths, mode=mode)
        eval_data = training_data_from_paths(self.eval_data_paths, mode=mode)
        test_data = training_data_from_paths(self.test_data_paths, mode=mode)
        return TrainingData(train_data=train_data,
                            eval_data=eval_data,
                            test_data=test_data,
                            **kwargs)

# if __name__ == '__main__':
#     a = DrugImporter(config_file="../../../configs/config.yml",
#                  train_data_paths="../../../datasets/train.csv",
#                  eval_data_paths="../../../datasets/eval.csv")
#     b = a.get_data(data_type="gen")
#
#     print(b)