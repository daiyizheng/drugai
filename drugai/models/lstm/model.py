#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 23:20
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : lstm.py
from __future__ import annotations, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.num_class = args.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=args.padding_ids)
        self.encoder = nn.LSTM(self.embed_size, self.hidden_size,
                            num_layers=args.num_layers, dropout=args.dropout, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.num_class)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=args.padding_ids)

    def forward(self,
                input_ids: torch.Tensor,
                lengths:torch.Tensor,
                hidden: torch.Tensor = None,
                target_ids:torch.Tensor=None):

        embed = self.embedding(input_ids)
        pack_pad_embed = pack_padded_sequence(embed, lengths=lengths.to("cpu"), batch_first=True)
        encoder, hidden = self.encoder(pack_pad_embed, hidden)
        pad_pack_encoder, _ = pad_packed_sequence(encoder, batch_first=True)
        logits = self.classifier(pad_pack_encoder)

        outputs = (logits, lengths, hidden)
        if target_ids is not None:
            loss = self.loss_func(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
            outputs += (loss,)

        return outputs


