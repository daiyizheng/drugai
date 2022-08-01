# -*- encoding: utf-8 -*-
'''
Filename         :model.py
Description      :
Time             :2022/08/01 22:26:29
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Text
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)



class VAE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 encoder_hidden_size: int,
                 encoder_num_layers: int,
                 encoder_bidirectional: bool,
                 encoder_z_liner_dim: int,
                 decoder_hidden_size: int,
                 decoder_num_layers: int,
                 decoder_bidirectional: bool,
                 decoder_z_liner_dim: int,
                 encodr_dropout_rate: float = 0.5,
                 decoder_dropout_arte: float = 0.5,
                 pad_token_ids: int = 0,
                 encoder_rnn_type: Text = "gru",
                 decoder_rnn_type: Text = "gru",
                 freeze_embeddings: bool = False
                 ):
        super(VAE, self).__init__()
        self.pad_token_ids = pad_token_ids

        vectors = torch.eye(vocab_size)
        embedding_dim = vocab_size
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
                dropout=encodr_dropout_rate if encoder_num_layers > 1 else 0,
                bidirectional=encoder_bidirectional
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_encoder_last = encoder_hidden_size * (2 if encoder_bidirectional else 1)
        self.q_mu = nn.Linear(q_encoder_last, encoder_z_liner_dim)
        self.q_logvar = nn.Linear(q_encoder_last, encoder_z_liner_dim)

        ##
        if decoder_rnn_type == "gru":
            self.decoder_rnn = nn.GRU(
                input_size=embedding_dim + encoder_z_liner_dim,
                hidden_size=decoder_hidden_size,
                num_layers=decoder_num_layers,
                batch_first=True,
                dropout=decoder_dropout_arte if decoder_num_layers > 1 else 0,
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

    def forward(self, input_ids: torch.Tensor):
        z, kl_loss = self.forward_encoder(input_ids=input_ids)
        recon_loss = self.forward_decoder(input_ids=input_ids, z=z)
        return kl_loss, recon_loss

    def forward_encoder(self,
                        input_ids: torch.Tensor):
        x = [self.embedding(x) for x in input_ids]
        x = nn.utils.rnn.pack_sequence(x)  # 自动padding
        _, h = self.encoder_rnn(x, None)  # [bz, seq_len, hz] [D∗num_layers,N,Hout​]
        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]  # (D∗num_layers,N,Hout​)
        h = torch.cat(h.split(1), dim=-1).squeeze(0)  # [N, D*num_layer*hz]
        mu, logvar = self.q_mu(h), self.q_logvar(h)  # 学习均值和方差
        eps = torch.randn_like(mu)  # 返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充
        z = mu + (logvar / 2).exp() * eps  # z分布~p(z)
        # 公式推导： log(\frac{\sigma_2}{\sigma_1}) + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
        # 假设N2是一个正态分布，也就是说\mu_2=0, \sigma_2^2=1, 也就是说KL(\mu_1, \sigma_1) = -log(\sigma_1) + \frac{\sigma_1^2+\mu_1^2}{2} - \frac{1}{2}
        # -->  kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # 参考来自： https://blog.csdn.net/qq_31895943/article/details/90754390
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()  # dl(p(z|x)||p(z))
        return z, kl_loss

    def forward_decoder(self,
                        input_ids: torch.Tensor,
                        z: torch.Tensor):
        lengths = [len(i_x) for i_x in input_ids]
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                              batch_first=True,
                                              padding_value=self.pad_token_ids)
        x_emb = self.embedding(input_ids)
        # 当参数只有两个时：（列的重复倍数，行的重复倍数）。1表示不重复  当参数有三个时：（通道数的重复倍数，列的重复倍数，行的重复倍数）。
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)
        h_0 = self.decoder_lat(z)  # decoder 部分 z 作为 h
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        output, _ = self.decoder_rnn(x_input, h_0)  # [num_layer*D, bz, hz]
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            input_ids[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_ids
        )  # MSE
        return recon_loss
