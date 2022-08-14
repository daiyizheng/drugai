#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 13:07
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : training_data_importer.py
from __future__ import annotations, print_function

from typing import Optional, Text, Union, List, Any, Dict
import logging, os
from abc import ABC, abstractmethod

from drugai.shared.training_data import TrainingData
import drugai.utils.common
from drugai.utils.io import read_config_yaml


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
    
    @staticmethod
    def load_from_config(config_file: Optional[Text] = None,
                         train_data_paths: Optional[Union[List[Text], Text]] = None,
                         eval_data_paths: Optional[Union[List[Text], Text]] = None,
                         test_data_paths: Optional[Union[List[Text], Text]] = None):
        if not os.path.exists(config_file):
            raise ValueError(f"Configuration file '{config_file}' does not exist.")
        config = read_config_yaml(config_file) ## 加载config yml文件
        return TrainingDataImporter.load_from_dict(config=config,
                                                  config_file=config_file,
                                                  train_data_paths=train_data_paths,
                                                  eval_data_paths=eval_data_paths,
                                                  test_data_paths=test_data_paths)

    @staticmethod
    def load_from_dict(config: Optional[Dict] = None,
                       config_file: Optional[Text] = None,
                       train_data_paths: Optional[Union[List[Text], Text]] = None,
                       eval_data_paths: Optional[Union[List[Text], Text]] = None,
                       test_data_paths: Optional[Union[List[Text], Text]] = None):
        from drugai.shared.importers.drug_importer import DrugImporter
        config = config or {}
        importers = config.get("importers", [])  ## 导入依赖,这里是自定义的导入模块

        if len(importers)>1:
            logging.warning("multiple `importer` modules are not supported")
        
        importers = [
            TrainingDataImporter._importer_from_dict(
                importer, config_file, train_data_paths, eval_data_paths, test_data_paths
            )
            for importer in importers
        ]
        if not importers:
            importers = [DrugImporter(config_file=config_file,
                                      train_data_paths=train_data_paths,
                                      eval_data_paths=eval_data_paths,
                                      test_data_paths=test_data_paths)]

        return importers[0]

    @staticmethod
    def _importer_from_dict(importer_config: Dict,
                            config_file: Text,
                            train_data_paths: Text, 
                            eval_data_paths: Text, 
                            test_data_paths: Text):
        from drugai.shared.importers.drug_importer import DrugImporter
        
        module_path = importer_config.pop("name", None)
        if module_path == DrugImporter.__name__:
            importer_class = DrugImporter
        else:
            try:
                importer_class = drugai.utils.common.class_from_module_path()
            except (AttributeError, ImportError):
                logging.warning(f"Importer '{module_path}' not found.")
                return None
        importer_config = dict(**importer_config)

        constructor_arguments = drugai.utils.common.minimal_kwargs(
            importer_config, importer_class
        )
        return importer_class(config_file=config_file, 
                              train_data_paths=train_data_paths, 
                              eval_data_paths=eval_data_paths,
                              test_data_paths=test_data_paths,
                               **constructor_arguments)

