#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 19:53
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : vae_gen.py
from __future__ import annotations, print_function
from typing import Text
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VAE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim:int,
                 encoder_hidden_size:int,
                 encoder_num_layers:int,
                 encoder_bidirectional:bool,
                 encoder_z_liner_dim:int,
                 decoder_hidden_size:int,
                 decoder_num_layers:int,
                 decoder_bidirectional: bool,
                 decoder_z_liner_dim: int,
                 encodr_dropout_rate:float=0.5,
                 decoder_dropout_arte:float=0.5,
                 pad_token_ids: int=0,
                 encoder_rnn_type:Text="gru",
                 decoder_rnn_type:Text="gru",
                 freeze_embeddings:bool=False
                 ):
        super(VAE, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_hidden_size =encoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_z_liner_dim = encoder_z_liner_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_layers = decoder_num_layers
        self.decoder_bidirectional = decoder_bidirectional
        self.decoder_z_liner_dim = decoder_z_liner_dim
        self.encodr_dropout_rate = encodr_dropout_rate
        self.decoder_dropout_arte = decoder_dropout_arte
        self.pad_token_ids = pad_token_ids

        vectors = torch.eye(vocab_size)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_ids)
        self.embedding.weight.data.copy_(vectors)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        if encoder_rnn_type == "gru":
            self.encoder_rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=encoder_hidden_size,
                num_layers=encoder_num_layers,
                batch_first=True,
                dropout=encodr_dropout_rate if encoder_num_layers>1 else 0,
                bidirectional=encoder_bidirectional
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_encoder_last = encoder_hidden_size*(2 if encoder_bidirectional else 1)
        self.q_mu = nn.Linear(q_encoder_last, encoder_z_liner_dim)
        self.q_logvar = nn.Linear(q_encoder_last, encoder_z_liner_dim)

        ##
        if decoder_rnn_type == "gru":
            self.decoder_rnn = nn.GRU(
                input_size=embedding_dim+encoder_z_liner_dim,
                hidden_size=decoder_hidden_size,
                num_layers=decoder_num_layers,
                batch_first=True,
                dropout=decoder_dropout_arte if decoder_num_layers>1 else 0,
                bidirectional=decoder_bidirectional
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )
        self.decoder_lat = nn.Linear(encoder_z_liner_dim, decoder_z_liner_dim)
        self.decoder_fc = nn.Linear(decoder_z_liner_dim, vocab_size)

        ## group

        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
            ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_lat,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.encoder,
            self.decoder
        ])

    def forward(self, input_ids:torch.Tensor):
        pass

    def forward_encoder(self,input_ids:torch.Tensor):
        pass

    def forward_decoder(self, input_ids:torch.Tensor, z:torch.Tensor):
        pass


