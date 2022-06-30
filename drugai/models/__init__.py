#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 15:34
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py
from drugai.models.gen_model.char_rnn.model import RNN as CharRNN
from drugai.models.gen_model.char_rnn.trainer import Trainer as CharTrainer
from drugai.models.dataset_base import DataSetBase
from drugai.models.vocab_base import Vocab

MODEL_CLASSES = {
    "char_rnn":(CharRNN, Vocab, CharTrainer, DataSetBase)
}
