#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 21:25
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trainer.py
from __future__ import annotations, print_function
import logging

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

from drugai.models.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, config):
        super(Trainer, self).__init__(config=config)

    def config_optimizer(self, *args, **kwargs):
        def get_params():
            return (p for p in self.model.parameters() if p.requires_grad)

        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.config.step_size, self.config.gamma)
        return optimizer, scheduler

    def get_train_dataloader(self):
        return self.dataset.train_dataloader()

    def train_epoch(self, *args, **kwargs):
        for step, batch_data in enumerate(self.epoch_data):
            self.train_step(batch_data, step)
        self.logs["learning_rate"] = self.scheduler.get_lr()[0]

    def train_step(self, batch_data, step):
        input_ids, target, lengths = batch_data
        batch = {
            "input_ids": input_ids.to(self.config.device),
            "lengths": lengths.to(self.config.device)
        }
        target = target.to(self.config.device)
        logits = self(batch)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), target.view(-1))
        if self.config.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        if self.config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.logs["loss"] += loss.item()

        self.epoch_data.set_postfix({"loss": self.logs["loss"] / step, **{"step": step}})
        self.global_step = 1

        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                torch.nn.utils.clip_grad_norm(amp.master_params(self.optimizer), self.config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
        return logits

    def get_evaluate_dataloader(self, *args, **kwargs):
        return self.dataset.eval_dataloader()

    def evaluate_epoch(self):
        preds_logits = None
        targets = None
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

    def evaluate_step(self, batch_data, step):
        input_ids, target, lengths = batch_data
        batch = {
            "input_ids": input_ids.to(self.config.device),
            "lengths": lengths.to(self.config.device)
        }
        target = target.to(self.config.device)
        logits = self(batch)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), target.view(-1))
        self.logs["eval_loss"] += loss.item()
        self.eval_data.set_postfix({**{"eval_loss": self.logs["eval_loss"] / step}, **{"eval_step": step}})
        return logits

    def get_predict_dataloader(self, *args, **kwargs):
        return self.dataset.test_dataloader()

    def predict(self, gen_batch_size, max_length):
        self.dataset.step(is_train=False, batch_size=gen_batch_size)
        test_dataloader = self.get_predict_dataloader()
        test_dataloader = test_dataloader.unsqueeze(1)
        self.model.eval()

        new_smiles_list = [torch.tensor(self.vocab.pad_token_ids, dtype=torch.long,
                                        device=self.config.device).repeat(max_length + 2)
                           for _ in range(gen_batch_size)]

        for i in range(gen_batch_size):
            new_smiles_list[i][0] = self.vocab.bos_token_ids

        len_smiles_list = [1 for _ in range(gen_batch_size)]
        lens = torch.tensor([1 for _ in range(gen_batch_size)], dtype=torch.long, device=self.config.device)
        end_smiles_list = [False for _ in range(gen_batch_size)]

        for i in range(1, max_length + 1):  # 列
            logits = self(input_ids = test_dataloader, lengths = lens, is_train=False)
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

            test_dataloader = torch.tensor(ind_tops, dtype=torch.long, device=self.config.device).unsqueeze(1)

        new_smiles_list = [new_smiles_list[i][:l] for i, l in enumerate(len_smiles_list)]
        return [self.vocab.ids_to_string(t) for t in new_smiles_list]


    def sample(self, *args, **kwargs):
        n_sample = kwargs.get("n_sample", None)
        if n_sample is None:
            raise KeyError

        samples = []
        n = n_sample
        with tqdm(n, desc="Generating sample") as T:
            while n_sample > 0:
                current_sample = self.predict(min(n, self.config.batch_size), self.config.max_length)
                samples.extend(current_sample)
                n_sample -= len(current_sample)
                T.update(len(current_sample))
        return samples

