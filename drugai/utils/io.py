#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 0:19
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : io.py
from __future__ import annotations, print_function

import errno
import json
from pathlib import Path
from typing import Text, Union, List, Any, Dict
import os
from ruamel import yaml as yaml

import pandas as pd
import numpy as np

from drugai.utils.constants import DEFAULT_ENCODING, DEFAULT_CSV_ENCODING

AVAILABLE_SPLITS = ['train', 'test', 'test_scaffolds']


def _is_ascii(text: Text) -> bool:
    return all(ord(character) < 128 for character in text)


def write_text_file(
        content: Text,
        file_path: Union[Text, Path],
        encoding: Text = DEFAULT_ENCODING,
        append: bool = False,
) -> None:
    mode = "a" if append else "w"
    with open(file_path, mode, encoding=encoding) as file:
        file.write(content)


def write_json_to_file(filename: Text, obj: Any, **kwargs: Any) -> None:
    """Write an object as a json string to a file."""

    write_to_file(filename, json_to_string(obj, **kwargs))


def write_to_file(filename: Text, text: Any) -> None:
    """Write a text to a file."""

    write_text_file(str(text), filename)


def write_dict_to_csv(
        content: Dict,
        file_path: Union[Text, Path],
        encoding: Text = DEFAULT_CSV_ENCODING,
) -> None:
    pd.DataFrame(content).to_csv(file_path,
                                 index=False,
                                 encoding=encoding)


def json_to_string(obj: Any,
                   **kwargs: Any) -> Text:
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def read_file(filename) -> Any:
    with open(filename, encoding="utf-8") as f:
        return f.read()


def read_yaml(content: Text,
              reader_type: Union[Text, List[Text]] = "safe"
              ) -> Any:
    if _is_ascii(content):
        # Required to make sure emojis are correctly parsed
        content = (
            content.encode("utf-8")
                .decode("raw_unicode_escape")
                .encode("utf-16", "surrogatepass")
                .decode("utf-16")
        )

    yaml_parser = yaml.YAML(typ=reader_type)
    yaml_parser.preserve_quotes = True

    return yaml_parser.load(content) or {}


def read_config_yaml(filename: Union[Text, Path]) -> Any:
    content = read_file(filename)
    return read_yaml(content)


def read_csv(path: Text):
    return pd.read_csv(path)


def read_smiles_csv(path: Text,
                    usecols: List = ["SMILES"]):
    return pd.read_csv(path,
                       usecols=usecols,
                       squeeze=True)


def read_smiles_zip(path: Text):
    return pd.read_csv(path,
                       compression='gzip')['SMILES'].values


def get_dataset(split='train'):
    """
    Arguments:
        split (str): split to load. Must be
            one of: 'train', 'test', 'test_scaffolds'
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}"
        )
    base_path = os.path.dirname(os.path.dirname(__file__))
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}")
    path = os.path.join(base_path, 'resources', split + '.csv.gz')
    smiles = pd.read_csv(path, compression='gzip')['SMILES'].values
    return smiles


def get_statistics(split='test'):
    """:arg
    code from moses:
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_path, 'resources', split + '_stats.npz')
    return np.load(path, allow_pickle=True)['stats'].item()


def create_directory(directory_path: Text) -> None:
    """Creates a directory and its super paths.

    Succeeds even if the path already exists."""

    try:
        os.makedirs(directory_path)
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise
