# -*- encoding: utf-8 -*-
'''
Filename         :rnn_generate.py
Description      :
Time             :2022/08/01 22:14:13
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Text, Any, Dict, Optional, List
import logging
import os
from functools import partial

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from drugai.shared.preprocess.basic_preprocessor import BasicPreprocessor
from .model import RNN
from drugai.models.dataset import default_collate_fn
from drugai.models.generate.gen_component import GenerateComponent
from drugai.models.generate.gen_vocab import CharRNNVocab, Vocab
from drugai.shared.importers.training_data_importer import TrainingDataImporter

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

logger = logging.getLogger(__name__)


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
        super(RNNGenerate, self).__init__(component_config=component_config, **kwargs)
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

    def train(self,
              file_importer: TrainingDataImporter,
              **kwargs):
        training_data = file_importer.get_data(preprocessor = BasicPreprocessor(),
                                               num_workers=kwargs.get("num_workers", None) \
                                                   if kwargs.get("num_workers", None) else 0)
        self.vocab = training_data.build_vocab(CharRNNVocab)
        self.component_config["vocab_size"] = len(self.vocab)
        self.component_config["pad_token_ids"] = self.vocab.pad_token_ids
        self.model = RNN(vocab_size=self.component_config["vocab_size"],
                         num_layers=self.component_config["num_layers"],
                         dropout_rate=self.component_config["dropout_rate"],
                         hidden_size=self.component_config["hidden_size"],
                         pad_token_ids=self.component_config["pad_token_ids"])
        self.model.to(self.device)
        train_dataloader = training_data.dataloader(batch_size=self.component_config["batch_size"],
                                                    collate_fn=partial(default_collate_fn, self.vocab),
                                                    shuffle=True,
                                                    mode="train")

        eval_dataloader = training_data.dataloader(batch_size=self.component_config["batch_size"],
                                                   collate_fn=partial(default_collate_fn, self.vocab),
                                                   shuffle=False,
                                                   mode="eval")

        self.optimizer, scheduler = self.config_optimizer()
        self.criterion = self.config_criterion()
        self.compute_metric = None
        self.model.zero_grad()

        for epoch in range(self.component_config["epochs"]):
            scheduler.step()  # Update learning rate schedule
            self.logs = {"loss": 0.0, "eval_loss": 0.0}
            self.epoch_data = tqdm(train_dataloader, desc='Training (epoch #{})'.format(epoch))
            self.model.train()
            self.train_epoch()
            self.logs["learning_rate"] = scheduler.get_lr()[0]
            if training_data.eval_data is not None:
                self.evaluate(eval_dataloader=eval_dataloader)
            for key, value in self.logs.items():
                self.tb_writer.add_scalar(key, value, epoch)

    def train_epoch(self, *args, **kwargs):
        for step, batch_data in enumerate(self.epoch_data):
            self.train_step(batch_data, step)

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

        self.logs["loss"] = (self.logs["loss"] + loss.item()) / (step + 1)
        self.epoch_data.set_postfix({"loss": self.logs["loss"], **{"step": step + 1}})
        self.global_step = 1

        if (step + 1) % self.component_config["gradient_accumulation_steps"] == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm(amp.master_params(self.optimizer), self.component_config["max_grad_norm"])
            else:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.component_config["max_grad_norm"])
        return logits

    def evaluate(self, eval_dataloader):
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

    @torch.no_grad()
    def evaluate_step(self, batch_data, step, **kwargs):
        input_ids, target, lengths = batch_data
        batch = {
            "input_ids": input_ids.to(self.device),
            "lengths": lengths.to(self.device)
        }
        target = target.to(self.device)
        logits, _ = self(**batch)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), target.view(-1))
        self.logs["eval_loss"] = (self.logs["eval_loss"] + loss.item()) / (step + 1)
        self.eval_data.set_postfix({**{"eval_loss": self.logs["eval_loss"]}, **{"eval_step": step + 1}})
        return logits

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

    def process(self,
                *args,
                **kwargs) -> Dict:
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
        model_state_dict = torch.load(os.path.join(model_dir, meta["name"] + "_model.pt"))
        model.load_state_dict(model_state_dict)
        vocab = torch.load(os.path.join(model_dir, meta["name"] + "_vocab.pt"))
        return cls(component_config=meta,
                   model=model,
                   vocab=vocab,
                   **kwargs)
