#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 13:46
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : rnn_generate.py
from __future__ import annotations, print_function

import argparse
import os
from functools import partial
from typing import Text, Any, Dict, Optional, List
import logging

from moses.script_utils import read_smiles_csv
from tqdm import tqdm
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from drugai.models.dataset import default_collate_fn
from drugai.models.generate.gen_vocab import CharRNNVocab
from drugai.models.vocab import Vocab
from drugai.models.generate.gen_component import GenerateComponent

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

logger = logging.getLogger(__name__)


class RNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 num_layers: int,
                 dropout_rate: int,
                 hidden_size: int,
                 pad_token_ids: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.vocab_size = self.input_size = self.output_size = vocab_size
        self.embedding_layer = nn.Embedding(self.vocab_size, self.vocab_size, padding_idx=pad_token_ids)
        self.lstm_layer = nn.LSTM(self.input_size,
                                  self.hidden_size,
                                  self.num_layers,
                                  dropout=self.dropout_rate,
                                  batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,
                input_ids: torch.Tensor,
                lengths: torch.Tensor,
                hiddens: torch.Tensor = None):
        embedd = self.embedding_layer(input_ids)
        embedd = pack_padded_sequence(embedd, lengths.to("cpu"), batch_first=True)
        embedd, hiddens = self.lstm_layer(embedd, hiddens)
        embedd, _ = pad_packed_sequence(embedd, batch_first=True)
        embedd = self.linear_layer(embedd)

        return embedd, hiddens


class RNNGenerate(GenerateComponent):
    defaults = {
        "vocab_size": -1,
        "num_layers": 3,
        "dropout_rate": 0.5,
        "hidden_size": 768,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.001,
        "max_length": 100,
        "batch_size": 512,
        "epochs": 10,
        "warmup_steps": 10,
        "gamma": 0.5,
        "pad_token_ids": -1
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 vocab: Vocab = None,
                 model=None,
                 **kwargs
                 ):
        super(RNNGenerate, self).__init__(component_config, **kwargs)
        ## vocab
        self.vocab = vocab
        self.model = model

    def config_optimizer(self, *args, **kwargs):
        def get_params():
            return (p for p in self.model.parameters() if p.requires_grad)

        optimizer = optim.Adam(get_params(), lr=self.component_config["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              self.component_config["warmup_steps"],
                                              self.component_config["gamma"])
        return optimizer, scheduler

    def config_criterion(self, *args, **kwargs):
        return nn.CrossEntropyLoss(ignore_index=self.component_config["pad_token_ids"])

    def load_data(self, file_dir: Text) -> np.ndarray:
        dataset = read_smiles_csv(file_dir)
        return dataset

    def build_vocab(self, dataset: np.ndarray) -> Vocab:
        return CharRNNVocab.from_data(dataset)

    def train(self,
              train_dir: Text,
              eval_dir: Text = None,
              **kwargs):
        ## 读取数据
        train_dataset = self.load_data(train_dir)

        self.vocab = self.build_vocab(train_dataset)
        self.component_config["vocab_size"] = len(self.vocab)
        self.component_config["pad_token_ids"] = self.vocab.pad_token_ids
        self.model = RNN(vocab_size=self.component_config["vocab_size"],
                         num_layers=self.component_config["num_layers"],
                         dropout_rate=self.component_config["dropout_rate"],
                         hidden_size=self.component_config["hidden_size"],
                         pad_token_ids=self.component_config["pad_token_ids"])
        self.model.to(self.device)
        train_dataloader = self.get_train_dataloader(train_dataset)

        self.optimizer, self.scheduler = self.config_optimizer()
        self.criterion = self.config_criterion()
        self.compute_metric = None
        self.model.zero_grad()

        for epoch in range(self.component_config["epochs"]):
            self.scheduler.step()  # Update learning rate schedule
            self.logs = {"loss": 0.0, "eval_loss": 0.0}
            self.epoch_data = tqdm(train_dataloader, desc='Training (epoch #{})'.format(epoch))
            self.model.train()
            self.train_epoch()
            if eval_dir is not None:
                self.evaluate(eval_dir)
            for key, value in self.logs.items():
                self.tb_writer.add_scalar(key, value, epoch)

    def train_epoch(self, *args, **kwargs):
        for step, batch_data in enumerate(self.epoch_data):
            self.train_step(batch_data, step)
        self.logs["learning_rate"] = self.scheduler.get_lr()[0]

    def train_step(self, batch_data, step):
        input_ids, target, lengths = batch_data
        batch = {
            "input_ids": input_ids.to(self.device),
            "lengths": lengths.to(self.device)
        }
        target = target.to(self.device)
        logits, _ = self(**batch)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), target.view(-1))

        self.optimizer.zero_grad()

        if self.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.component_config["gradient_accumulation_steps"] > 1:
            loss = loss / self.component_config["gradient_accumulation_steps"]

        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        self.logs["loss"] += loss.item()
        self.epoch_data.set_postfix({"loss": self.logs["loss"] / (step + 1), **{"step": step + 1}})
        self.global_step = 1

        if (step + 1) % self.component_config["gradient_accumulation_steps"] == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm(amp.master_params(self.optimizer), self.component_config["max_grad_norm"])
            else:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.component_config["max_grad_norm"])
        return logits

    def evaluate(self, eval_dir: Text):
        eval_dataset = self.load_data(eval_dir)
        eval_dataloader = self.get_evaluate_dataloader(eval_dataset)

        self.eval_data = tqdm(eval_dataloader, desc='Evaluation')
        self.evaluate_epoch()

    def evaluate_epoch(self, *args, **kwargs):
        preds_logits = None
        targets = None
        self.model.eval()

        for step, batch_data in enumerate(self.eval_data):
            logits = self.evaluate_step(batch_data, step)
            if self.compute_metric is not None:
                _, target, _ = batch_data
                if preds_logits is None:
                    preds_logits = logits
                    targets = target
                else:
                    preds_logits = np.append(preds_logits, logits.detach().cpu().numpy(), axis=0)
                    targets = np.append(targets, target.detach().cpu().numpy(), axis=0)

        if self.compute_metric is not None:
            result = self.compute_metric(preds_logits, targets)
            for key in result.keys():
                self.logs["eval_" + key] = result[key]

    def evaluate_step(self, batch_data, step, **kwargs):
        input_ids, target, lengths = batch_data
        batch = {
            "input_ids": input_ids.to(self.device),
            "lengths": lengths.to(self.device)
        }
        target = target.to(self.device)
        logits, _ = self(**batch)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), target.view(-1))
        self.logs["eval_loss"] += loss.item()
        self.eval_data.set_postfix({**{"eval_loss": self.logs["eval_loss"] / (step + 1)}, **{"eval_step": step + 1}})
        return logits

    def get_train_dataloader(self, dataset, **kwargs):
        return DataLoader(dataset,
                          batch_size=self.component_config["batch_size"],
                          shuffle=True,
                          collate_fn=partial(default_collate_fn, self.vocab))

    def get_evaluate_dataloader(self, dataset, **kwargs):
        return DataLoader(dataset,
                          batch_size=self.component_config["batch_size"],
                          shuffle=False,
                          collate_fn=partial(default_collate_fn, self.vocab))

    def get_predict_dataloader(self, *args, **kwargs):
        dataset = [torch.tensor([self.vocab.bos_token_ids],
                                dtype=torch.long) for _ in range(self.component_config["batch_size"])]
        dataset = torch.tensor(dataset, dtype=torch.long)
        dataset = dataset.unsqueeze(1)
        return DataLoader(dataset,
                          batch_size=self.component_config["batch_size"],
                          shuffle=False,
                          collate_fn=None)

    @torch.no_grad()
    def predict(self,
                batch_size: int,
                max_length: int,
                **kwargs
                ) -> List[str]:

        test_dataloader = self.get_predict_dataloader()
        test_dataloader = next(iter(test_dataloader)).to(self.device)
        self.model.eval()

        new_smiles_list = [torch.tensor(self.vocab.pad_token_ids, dtype=torch.long,
                                        device=self.device).repeat(max_length + 2)
                           for _ in range(batch_size)]

        for i in range(batch_size):
            new_smiles_list[i][0] = self.vocab.bos_token_ids

        len_smiles_list = [1 for _ in range(batch_size)]
        lens = torch.tensor([1 for _ in range(batch_size)], dtype=torch.long, device=self.device)
        end_smiles_list = [False for _ in range(batch_size)]

        hiddens = None
        for i in range(1, max_length + 1):  # 列
            logits, hiddens = self(input_ids=test_dataloader, lengths=lens, hiddens=hiddens)
            probs = [F.softmax(o, dim=-1) for o in logits]
            # sample from probabilities 按照概率采样
            ind_tops = [torch.multinomial(p, 1) for p in probs]

            for j, atom in enumerate(ind_tops):  ##行
                if not end_smiles_list[j]:
                    atom_elem = atom[0].item()
                    if atom_elem == self.vocab.eos_token_ids:
                        end_smiles_list[j] = True

                    new_smiles_list[j][i] = atom_elem
                    len_smiles_list[j] = len_smiles_list[j] + 1

            test_dataloader = torch.tensor(ind_tops, dtype=torch.long, device=self.device).unsqueeze(1)

        new_smiles_list = [new_smiles_list[i][:l] for i, l in enumerate(len_smiles_list)]
        return [self.vocab.ids_to_string(t) for t in new_smiles_list]

    def process(self, *args, **kwargs) -> Dict:
        n_sample = self.component_config["n_sample"]
        batch_size = self.component_config["batch_size"]
        max_length = self.component_config["max_length"]
        self.model.to(self.device)

        samples = []
        n = n_sample
        with tqdm(n, desc="Generating sample") as T:
            while n_sample > 0:
                current_sample = self.predict(min(n, batch_size), max_length)
                samples.extend(current_sample)
                n_sample -= len(current_sample)
                T.update(len(current_sample))
        return {"SMILES": samples}

    def persist(self, model_dir: Text
                ) -> Optional[Dict[Text, Any]]:
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        ## 保存模型
        torch.save(model_to_save.state_dict(), os.path.join(model_dir, self.name + "_model.pt"))
        ## 保存字典
        torch.save(self.vocab, os.path.join(model_dir, self.name + "_vocab.pt"))
        ## 保存参数
        torch.save(self.component_config, os.path.join(model_dir, self.name + "_component_config.pt"))
        return {"vocab_file": os.path.join(model_dir, self.name+"_vocab.pt"),
                "model_file": os.path.join(model_dir, self.name+"_model.pt"),
                "component_config": os.path.join(model_dir, self.name+"_component_config.pt")}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             **kwargs: Any
             ) -> "Component":
        model = RNN(vocab_size=meta["vocab_size"],
                    num_layers=meta["num_layers"],
                    dropout_rate=meta["dropout_rate"],
                    hidden_size=meta["hidden_size"],
                    pad_token_ids=meta["pad_token_ids"])
        model_state_dict = torch.load(os.path.join(model_dir, meta["name"]+"_model.pt"))
        model.load_state_dict(model_state_dict)
        vocab = torch.load(os.path.join(model_dir, meta["name"]+"_vocab.pt"))
        return cls(component_config=meta,
                   model=model,
                   vocab=vocab,
                   **kwargs)
