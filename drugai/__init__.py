#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 20:51
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py
from __future__ import annotations, print_function

from drugai.dataloader import Processor
from drugai.models.lstm.model import LSTMModel
from drugai.models.lstm.trainer import Trainer
from drugai.models.lstm.vocab import LstmVocab

MODEL_CLASSES = {
    "lstm": (LSTMModel, LstmVocab, Trainer, Processor)
}