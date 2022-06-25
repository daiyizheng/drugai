#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 15:33
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : trainer.py
from __future__ import annotations, print_function

import logging
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup
from torch.utils.data import DataLoader

from drugai.models.trainer_base import TrainerBase

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

logger = logging.getLogger(__file__)


class Trainer(TrainerBase):
    def __init__(self,
                 model,
                 vocab,
                 args,
                 collate_fn=None,
                 train_dataset=None,
                 eval_dataset=None,
                 test_dataset=None,
                 *arg,
                 **kwargs):
        super(Trainer, self).__init__(model=model,
                                      vocab=vocab,
                                      args=args,
                                      train_dataset=train_dataset,
                                      eval_dataset=eval_dataset,
                                      test_dataset=test_dataset,
                                      collate_fn=collate_fn)

    def train(self):
        # tensorboardx
        self.tb_writer = SummaryWriter(self.args.tensorboardx_path)

        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        train_dataloader = self.get_train_dataloader(self.train_dataset)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        optimizer, scheduler = self.config_optimizer(t_total)
        if self.args.fp16:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank],
                                                                   output_device=self.args.local_rank,
                                                                   find_unused_parameters=True)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.args.train_batch_size * self.args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if self.args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        self.model.zero_grad()
        global_step = 1
        logs = {"loss": 0.0, "running_loss": 0.0}
        for epoch in range(self.args.num_train_epochs):
            tqdm_data = tqdm(train_dataloader, desc='Training (epoch #{})'.format(epoch))
            self.model.train()
            for step, batch_data in enumerate(tqdm_data):
                logits, _, _, loss = self.forward(batch_data)

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                logs['loss'] = loss.item()
                logs['running_loss'] += (loss.item() - logs['running_loss']) / (step + 1)
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    tqdm_data.set_postfix({**logs, **{"step": global_step}})

            if self.args.local_rank == -1 and self.args.evaluate_during_training:
                results = self.evaluate()
                eval_loss = results["loss"]
                logger.info("*" * 50)
                logger.info("current step loss for logging steps: {}".format(eval_loss))
                logger.info("*" * 50)

                for key, value in results.items():
                    eval_key = "eval_{}".format(key)
                    logs[eval_key] = value
                    print("eval_{}".format(key), value, epoch)

            learning_rate_scalar = scheduler.get_lr()[0]
            logs["learning_rate"] = learning_rate_scalar
            for key, value in logs.items():
                self.tb_writer.add_scalar(key, value, epoch)
            ## 保存模型
            self.save_model()
            tqdm_data.set_postfix({**logs, **{"step": global_step}})

    def forward(self, batch_data, is_train=True):
        if is_train:
            source, target, lengths = batch_data
            target = target.to(self.args.device)
        else:
            source, lengths = batch_data
            target = None
        source = source.to(self.args.device)
        lengths = lengths.to(self.args.device)

        output = self.model(input_ids=source, lengths=lengths, hidden=None, target_ids=target)
        return output

    def config_optimizer(self, total=None):
        # Prepare optimizer and schedule (linear warmup and decay)
        print("**********************************Prepare optimizer and schedule start************************")
        for n, p in self.model.named_parameters():
            print(n)
        print("**********************************Prepare optimizer and schedule middle************************")
        optimizer_grouped_parameters = []
        # embedding部分
        embeddings_params = list(self.model.embedding.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        if not self.args.embeddings_learning_rate:
            self.args.embeddings_learning_rate = self.args.learning_rate
        optimizer_grouped_parameters += [
            {'params': [p for n, p in embeddings_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.embeddings_learning_rate,
             },
            {'params': [p for n, p in embeddings_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.embeddings_learning_rate,
             }
        ]
        # encoder + decoder 部分
        encoder_params = list(self.model.encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.lstm_learning_rate,
             },
            {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.lstm_learning_rate,
             }
        ]
        # linear层
        classifier_params = list(self.model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.classifier_learning_rate,
             },
            {'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.classifier_learning_rate,
             },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total,
            power=2)
        return optimizer, scheduler

    @torch.no_grad()
    def evaluate(self):
        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_dataloader = self.get_evaluate_dataloader(self.eval_dataset)

        # Eval!
        logger.info("***** Running evaluation on dev dataset *****")
        logger.info("  Num examples = %d", len(eval_dataloader))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        tqdm_data = tqdm(eval_dataloader, desc='Evaluation')

        eval_loss = 0.0
        nb_eval_steps = 0
        logs = {"eval_loss": 0.0, "eval_running_loss": 0.0}
        for step, batch_data in enumerate(tqdm_data):
            _, _, hidden, loss = self.forward(batch_data)
            eval_loss += loss.item()
            nb_eval_steps += 1
            logs['eval_loss'] = loss.item()
            logs['eval_running_loss'] += (loss.item() - logs['eval_running_loss']) / (step + 1)
            tqdm_data.set_postfix({**logs, **{"eval_step": step}})
        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        return results

    @torch.no_grad()
    def predict(self,gen_number):
        self.args.test_batch_size = min(self.args.per_gpu_test_batch_size * max(1, self.args.n_gpu), gen_number)
        test_dataloader = self.get_predict_dataloader(self.test_dataset)
        test_dataloader = test_dataloader.unsqueeze(1)
        self.model.eval()
        new_smiles_list = [torch.tensor(self.vocab.pad_token_ids, dtype=torch.long, device=self.args.device).repeat(
            self.args.max_length + 2) for _ in range(self.args.test_batch_size)]

        for i in range(self.args.test_batch_size):
            new_smiles_list[i][0] = self.vocab.bos_token_ids
        len_smiles_list = [1 for _ in range(self.args.test_batch_size)]
        lens = torch.tensor([1 for _ in range(self.args.test_batch_size)], dtype=torch.long, device=self.args.device)
        end_smiles_list = [False for _ in range(self.args.test_batch_size)]
        for i in range(1, self.args.max_length + 1):  # 列
            logits, _, _ = self.forward(batch_data = (test_dataloader, lens), is_train=False)
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

            test_dataloader = torch.tensor(ind_tops, dtype=torch.long, device=self.args.device).unsqueeze(1)

        new_smiles_list = [new_smiles_list[i][:l] for i, l in enumerate(len_smiles_list)]
        return [self.vocab.ids_to_string(t) for t in new_smiles_list]

    def get_train_dataloader(self, dataset=None):
        train_loader = DataLoader(dataset,
                                  batch_size=self.args.train_batch_size,
                                  shuffle=True,
                                  num_workers=self.args.n_workers,
                                  collate_fn=self.collate_fn)
        return train_loader

    def get_evaluate_dataloader(self, dataset=None):
        eval_loader = DataLoader(dataset,
                                 batch_size=self.args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=self.args.n_workers,
                                 collate_fn=self.collate_fn)
        return eval_loader

    def get_predict_dataloader(self, dataset=None):
        test_dataloader = [torch.tensor([self.vocab.bos_token_ids],
                                        dtype=torch.long,
                                        device=self.args.device
                                        ) for _ in range(self.args.test_batch_size)]

        test_dataloader = torch.tensor(test_dataloader,
                                       dtype=torch.long,
                                       device=self.args.device)

        return test_dataloader
