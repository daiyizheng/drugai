# -*- encoding: utf-8 -*-
'''
Filename         :trainer.py
Description      :
Time             :2022/08/02 00:28:29
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import annotations, print_function
import logging
from typing import Optional, Text, Dict, Any

from tqdm import tqdm

from drugai.models.generate.gen_component import GenerateComponent
from drugai.shared.importers.training_data_importer import TrainingDataImporter

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

logger = logging.getLogger(__name__)



class MoFlowGenerate(GenerateComponent):
    defaults = {
        "epochs": 5000,
        "batch_size": 512,


        "b_n_type": 4,
        "b_n_flow": 10,
        "b_n_block": 1,
        "b_n_squeeze": 3, # 3 or 2
        "b_hidden_ch": [128,128],
        "b_affine": True,
        "b_conv_lu": 1,
        "a_n_node": 9,
        "a_n_type": 5,
        "a_hidden_gnn": [64],
        "a_hidden_lin": [128, 64],
        "a_n_flow": 27,
        "a_n_block": 1,
        "mask_row_size_list": [1],
        "mask_row_stride_list": [1],
        "a_affine": True

    }

    def __init__(self, 
                 component_config: Optional[Dict[Text, Any]] = None,
                 model=None,
                 **kwargs):
        super().__init__(component_config=component_config, **kwargs)

        self.model = model

    def config_optimizer(self, *args, **kwargs) -> Any:
        pass

    def config_criterion(self, *args, **kwargs) -> Any:
        pass

    def train(self,
              file_importer: TrainingDataImporter,
              **kwargs) -> Any:
        pass

