#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/8/4 7:45 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : preprocessor.py
# @desc :
from __future__ import annotations, print_function
import logging
import abc
from typing import List, Text, Any

logger = logging.getLogger(__name__)


class Preprocessor(abc.ABC):
    @abc.abstractmethod
    def __init__(self,
                 *args,
                 **kwargs
                 ) -> None:
        ...
    
    @abc.abstractmethod
    def pre_process(self, data:List[Any]):
        ...

    @abc.abstractmethod
    def get_data_from_paths(self,
                            path:Text,
                            **kwargs
                            ) -> Any:
        ...

