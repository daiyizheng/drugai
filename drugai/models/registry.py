#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 22:05
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : registry.py
from __future__ import annotations, print_function

import logging
import traceback
from typing import Dict, Text, Any, Optional, Type

from drugai.models.component import Component
from drugai.models.generate.rnn_gen import RNNGenerate
from drugai.utils.common import class_from_module_path

logger = logging.getLogger(__name__)

component_classes = [
    RNNGenerate
]
registered_components = {c.name: c for c in component_classes}


def get_component_class(component_name: Text):
    if component_name not in registered_components:
        try:
            return class_from_module_path(component_name)
        except (ImportError, AttributeError) as e:
            is_path = "." in component_name

            if is_path:
                module_name, _, class_name = component_name.rpartition(".")
                if isinstance(e, ImportError):
                    exception_message = f"Failed to find module '{module_name}'."
                else:
                    # when component_name is a path to a class but the path does
                    # not contain that class
                    exception_message = (
                        f"The class '{class_name}' could not be "
                        f"found in module '{module_name}'."
                    )
            else:
                exception_message = (
                    f"Cannot find class '{component_name}' in global namespace. "
                    f"Please check that there is no typo in the class "
                    f"name and that you have imported the class into the global "
                    f"namespace."
                )
            raise ModuleNotFoundError(
                f"Failed to load the component "
                f"'{component_name}'. "
                f"{exception_message} Either your "
                f"pipeline configuration contains an error "
                f"or the module you are trying to import "
                f"is broken (e.g. the module is trying "
                f"to import a package that is not "
                f"installed). {traceback.format_exc()}"
            )

    return registered_components[component_name]


def load_component_by_meta(component_meta: Dict[Text, Any],
                           model_dir: Text,
                           **kwargs: Any,
                           ) -> Optional["Component"]:
    component_name = component_meta.get("class", component_meta["name"])
    component_class = get_component_class(component_name)
    return component_class.load(component_meta, model_dir, **kwargs)


def create_component_by_config(component_config: Dict[Text, Any],
                               **kwargs
                               ) -> "Component":
    component_name = component_config.get("class", component_config["name"])
    component_class = get_component_class(component_name)
    return component_class.create(component_config, **kwargs)

# if __name__ == '__main__':
#     config = {'name': 'RNNGenerate', 'batch_size': 512}
#     a = create_component_by_config(config)
#     print(a)
