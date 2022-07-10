#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/25 20:08
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : component.py
from __future__ import annotations
from typing import Text, Optional, Any, Dict, List


from drugai.utils.common import override_defaults


class ComponentMetaclass(type):
    """Metaclass with `name` class property."""

    @property
    def name(cls):
        """The name property is a function of the class - its __name__."""

        return cls.__name__


class Component(metaclass=ComponentMetaclass):
    defaults = {}

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs:Any
                 ) -> None:
        if not component_config:
            component_config = {}

        # makes sure the name of the configuration is part of the config
        # this is important for e.g. persistence
        component_config["name"] = self.name
        self.component_config = override_defaults(
            self.defaults, component_config
        )

    @property
    def name(self) -> Text:
        """Access the class's property name from an instance."""

        return type(self).name

    @classmethod
    def required_packages(cls) -> List[Text]:
        return []

    @classmethod
    def create(cls,
               component_config: Dict[Text, Any],
               **kwargs:Any
               ) -> "Component":
        return cls(component_config, **kwargs)

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             **kwargs:Any
             ) -> "Component":
        return cls(**kwargs)

    def process(self, *args, **kwargs: Any
                ) -> None:
        pass

    def train(self, *args, **kwargs: Any
              ) -> None:
        pass

    def persist(self, model_dir: Text
                ) -> Optional[Dict[Text, Any]]:
        pass













