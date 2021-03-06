#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 13:07
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : training_data_importer.py
from __future__ import annotations, print_function

from abc import ABC, abstractmethod
from typing import Optional, Text, Union, List, Any, Dict

from drugai.shared.training_data import TrainingData


class TrainingDataImporter(ABC):
    @abstractmethod
    def __init__(
            self,
            config_file: Optional[Text] = None,
            train_data_paths: Optional[Union[List[Text], Text]] = None,
            eval_data_paths: Optional[Union[List[Text], Text]] = None,
            test_data_paths: Optional[Union[List[Text], Text]] = None,
            **kwargs: Any,
    ) -> None:
        self.config_file = config_file
        self.train_data_paths = train_data_paths
        self.eval_data_paths = eval_data_paths
        self.test_data_paths = test_data_paths

    @abstractmethod
    def get_config(self) -> Dict:
        """Retrieves the configuration that should be used for the training.
        Returns:
            The configuration as dictionary.
        """
        ...

    @abstractmethod
    def get_data(self, **kwargs) -> TrainingData:
        """Retrieves the training data that should be used for training.
        Returns:
            Loaded data `TrainingData`.
        """
        ...
