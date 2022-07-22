#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 19:53
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : vae_gen.py
from __future__ import annotations, print_function

from functools import partial
from typing import Text, Optional, Any, Dict, List
import logging

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from moses.script_utils import read_smiles_csv

from drugai.models.dataset import single_collate_fn
from drugai.models.losses.cosine_annealing_lr_with_restart import CosineAnnealingLRWithRestart
from drugai.models.losses.kl_annealer import KLAnnealer
from drugai.models.generate.gen_component import GenerateComponent
from drugai.models.generate.gen_vocab import CharRNNVocab
from drugai.models.vocab import Vocab

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
        x_emb = self.x_emb(input_ids)
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


class CircularBuffer:
    def __init__(self, size):
        self.max_size = size
        self.data = np.zeros(self.max_size)
        self.size = 0
        self.pointer = -1

    def add(self, element):
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        self.data[self.pointer] = element
        return element

    def last(self):
        assert self.pointer != -1, "Can't get an element from an empty buffer!"
        return self.data[self.pointer]

    def mean(self):
        if self.size > 0:
            return self.data[:self.size].mean()
        return 0.0


class VAEGenerate(GenerateComponent):
    defaults = {
        "epochs": 10,
        "pad_token_ids": -1,
        "batch_size": 512,
        "max_length": 100,

        "vocab_size": -1,
        "encoder_hidden_size": 256,
        "encoder_num_layers": 1,
        "encoder_bidirectional": False,
        "encoder_z_liner_dim": 128,
        "decoder_hidden_size": 512,
        "decoder_num_layers": 3,
        "decoder_bidirectional": False,
        "decoder_z_liner_dim": 128,
        "encodr_dropout_rate": 0.5,
        "decoder_dropout_arte": 0,
        "encoder_rnn_type": "gru",
        "decoder_rnn_type": "gru",
        "freeze_embeddings": False,

        "clip_grad": 50,
        "kl_start": 0,
        "kl_w_start": 0.0,
        "kl_w_end": 0.05,
        "lr_start": 3 * 1e-4,
        "lr_n_period": 10,
        "lr_n_restarts": 10,
        "lr_n_mult": 1,
        "lr_end": 3 * 1e-4,
        "n_last": 1000,
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 vocab: Vocab = None,
                 model=None,
                 **kwargs
                 ):
        super(VAEGenerate, self).__init__(component_config=component_config)
        ## vocab
        self.vocab = vocab
        self.model = model

    def config_optimizer(self,
                         *args,
                         **kwargs):
        def get_optim_params(model):
            return (p for p in model.vae.parameters() if p.requires_grad)

        optimizer = optim.Adam(get_optim_params(self.model), lr=self.component_config["lr_start"])
        kl_annealer = KLAnnealer(epochs=self.component_config["epochs"],
                                 kl_start=self.component_config["kl_start"],
                                 kl_w_start=self.component_config["kl_w_start"],
                                 kl_w_end=self.component_config["kl_w_end"])  # 每个epoch来重置权重
        lr_annealer = CosineAnnealingLRWithRestart(optimizer=optimizer,
                                                   lr_n_period=self.component_config["lr_n_period"],
                                                   lr_n_mult=self.component_config["lr_n_mult"],
                                                   lr_end=self.component_config["lr_end"])
        return optimizer, [kl_annealer, lr_annealer]

    def config_criterion(self,
                         *args,
                         **kwargs):
        pass

    def load_data(self,
                  file_dir: Text
                  ) -> np.ndarray:
        dataset = read_smiles_csv(file_dir)
        return dataset

    def build_vocab(self,
                    dataset: np.ndarray
                    ) -> Vocab:
        return CharRNNVocab.from_data(dataset)

    def train(self,
              train_dir: Text,
              eval_dir: Text = None,
              **kwargs):
        ## 加载数据
        train_dataset = self.load_data(file_dir=train_dir)
        ## 加载字典
        self.vocab = self.build_vocab(train_dataset)
        self.component_config["vocab_size"] = len(self.vocab)
        self.component_config["pad_token_ids"] = self.vocab.pad_token_ids
        self.model = VAE(vocab_size=self.component_config["vocab_size"],
                         encoder_hidden_size=self.component_config["encoder_hidden_size"],
                         encoder_num_layers=self.component_config["encoder_num_layers"],
                         encoder_bidirectional=self.component_config["encoder_bidirectional"],
                         encoder_z_liner_dim=self.component_config["encoder_z_liner_dim"],
                         decoder_hidden_size=self.component_config["decoder_hidden_size"],
                         decoder_num_layers=self.component_config["decoder_num_layers"],
                         decoder_bidirectional=self.component_config["decoder_bidirectional"],
                         decoder_z_liner_dim=self.component_config["decoder_z_liner_dim"],
                         encodr_dropout_rate=self.component_config["encodr_dropout_rate"],
                         decoder_dropout_arte=self.component_config["decoder_dropout_arte"],
                         pad_token_ids=self.component_config["pad_token_ids"],
                         encoder_rnn_type=self.component_config["encoder_rnn_type"],
                         decoder_rnn_type=self.component_config["decoder_rnn_type"],
                         freeze_embeddings=self.component_config["freeze_embeddings"])
        self.model.to(self.device)
        train_dataloader = self.get_train_dataloader(train_dataset)

        self.optimizer, [kl_annealer, lr_annealer] = self.config_optimizer()
        self.compute_metric = None
        self.model.zero_grad()

        for epoch in range(self.component_config["epochs"]):
            kl_weight = kl_annealer(epoch)
            lr_annealer.step()
            self.logs = {"loss": 0.0, "eval_loss": 0.0}
            self.epoch_data = tqdm(train_dataloader, desc='Training (epoch #{})'.format(epoch))
            self.model.train()
            self.train_epoch()

            self.logs["learning_rate"] = lr_annealer.get_lr()[0]
            if eval_dir is not None:
                self.evaluate(eval_dir)
            for key, value in self.logs.items():
                self.tb_writer.add_scalar(key, value, epoch)


    def train_epoch(self,
                    *args,
                    **kwargs):
        pass

    def train_step(self,
                   batch_data,
                   step):
        pass

    def evaluate(self,
                 eval_dir: Text):
        pass

    def evaluate_epoch(self,
                       *args,
                       **kwargs):
        pass

    def evaluate_step(self,
                      batch_data,
                      step,
                      **kwargs):
        pass

    def get_train_dataloader(self,
                             dataset,
                             **kwargs):
        return DataLoader(dataset,
                          batch_size=self.component_config["batch_size"],
                          shuffle=True,
                          collate_fn=partial(single_collate_fn, self.vocab))

    def get_evaluate_dataloader(self,
                                dataset,
                                **kwargs):
        pass

    def get_predict_dataloader(self,
                               *args,
                               **kwargs):
        pass

    @torch.no_grad()
    def predict(self,
                batch_size: int,
                max_length: int,
                **kwargs
                ) -> List[str]:
        pass

    def process(self,
                *args,
                **kwargs) -> Dict:
        pass

    def persist(self, model_dir: Text
                ) -> Optional[Dict[Text, Any]]:
        pass

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             **kwargs: Any
             ) -> "Component":
        pass
