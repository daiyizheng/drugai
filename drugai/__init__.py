#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 20:51
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : __init__.py
from __future__ import annotations, print_function

from drugai.dataloader import Processor
from drugai.models.char_rnn.model import RNN
from drugai.models.char_rnn.trainer import Trainer as CharTrainer
from drugai.models.lstm.model import LSTMModel
from drugai.models.lstm.trainer import Trainer as LstmTrainer
from drugai.models.lstm.vocab import LstmVocab
from drugai.models.vocab_base import Vocab

MODEL_CLASSES = {
    "lstm": (LSTMModel, LstmVocab, LstmTrainer, Processor),
    "char_rnn":(RNN, Vocab, CharTrainer)
}