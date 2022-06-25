#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 22:33
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : vocab.py
from __future__ import annotations, print_function

import abc

import torch


class SpecialTokens:
    bos_token = '<BOS>'
    eos_token = '<EOS>'
    pad_token = '<PAD>'
    unk_token = '<UNK>'


class Vocab(abc.ABC):
    def __init__(self, data, special_token=SpecialTokens):
        if (special_token.bos_token in data) or (special_token.eos_token in data) or (
                special_token.pad_token in data) or (special_token.unk_token in data):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(data))+ [special_token.pad_token,
                                        special_token.unk_token,
                                        special_token.bos_token,
                                        special_token.eos_token]

        self.special_token = special_token
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos_token_ids(self):
        return self.c2i[self.special_token.bos_token]

    @property
    def bos_token(self):
        return self.special_token.bos_token

    @property
    def eos_token_ids(self):
        return self.c2i[self.special_token.eos_token]

    @property
    def eos_token(self):
        return self.special_token.eos_token

    @property
    def unk_token_ids(self):
        return self.c2i[self.special_token.unk_token]

    @property
    def unk_token(self):
        return self.special_token.unk_token

    @property
    def pad_token_ids(self):
        return self.c2i[self.special_token.pad_token]

    @property
    def pad_token(self):
        return self.special_token.pad_token

    def covert_token_to_ids(self, token):
        return self.c2i.get(token, self.unk_token_ids)

    def covert_ids_to_token(self, ids):
        return self.i2c.get(ids, self.unk_token)

    def string_to_ids(self, string, is_add_bos_eos_token_ids=True):
        token_ids = [self.covert_token_to_ids(s) for s in string]
        if is_add_bos_eos_token_ids:
            token_ids = [self.bos_token_ids] + token_ids + [self.eos_token_ids]
        return token_ids

    def ids_to_string(self, ids, is_del_bos_eos_token=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if is_del_bos_eos_token:
            ids = ids[1:]
            ids = ids[:-1]
        return "".join([self.covert_ids_to_token(i) for i in ids])

    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)

        return cls(chars, *args, **kwargs)