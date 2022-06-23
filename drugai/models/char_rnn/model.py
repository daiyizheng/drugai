#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 21:16
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : model.py
from __future__ import annotations, print_function

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharRNN(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 token_pad_ids: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout_rate: float):
        super(CharRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout_rate
        self.vocab_size = self.input_size = self.output_size = vocab_size

        self.embedding_layer = nn.Embedding(self.vocab_size, self.vocab_size,
                                            padding_idx=token_pad_ids)
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size,
                                  self.num_layers, dropout=self.dropout,
                                  batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size,
                                      self.output_size)

    def forward(self,
                input_ids: torch.Tensor,
                lengths: torch.Tensor,
                hiddens: torch.Tensor = None) -> Tuple:
        embedd = self.embedding_layer(input_ids)
        embedd = pack_padded_sequence(embedd, lengths.to("cpu"), batch_first=True)
        embedd, hiddens = self.lstm_layer(embedd, hiddens)
        embedd, _ = pad_packed_sequence(embedd, batch_first=True)
        embedd = self.linear_layer(embedd)

        return embedd, lengths, hiddens
