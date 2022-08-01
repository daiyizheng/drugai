# -*- encoding: utf-8 -*-
'''
Filename         :moflow_gene.py
Description      :
Time             :2022/07/31 16:47:29
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
from __future__ import annotations, print_function
from ast import Dict

import logging
from typing import Optional, Text
from anyio import Any

import torch.nn as nn

from drugai.models.generate.gen_component import GenerateComponent

logger = logging.getLogger(__name__)

class MoFlow(nn.Module):
    def __init__(self) -> None:
        super(MoFlow, self).__init__()
    


class MoFlowGenerate(GenerateComponent):
    defaults = {

    }

    def __init__(self, 
                 component_config: Optional[Dict[Text, Any]] = None,
                 model=None,
                 **kwargs):
        super().__init__(component_config=component_config, **kwargs)

        self.model = model

    