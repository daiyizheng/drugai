#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 0:11
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : metric_util.py
from __future__ import annotations, print_function
import argparse
import os
from typing import Text, List

from multiprocessing import Pool

import numpy as np
from moses.metrics import get_all_metrics, fraction_valid, remove_invalid, fraction_unique, \
    compute_intermediate_statistics, SNNMetric, ScafMetric, FragMetric, FCDMetric, internal_diversity, \
    fraction_passes_filters, logP, SA, weight, QED, WassersteinMetric
from moses.metrics.metrics import novelty
from moses.script_utils import read_smiles_csv
from moses.utils import disable_rdkit_log, get_mol, mapper, enable_rdkit_log

from drugai.utils.io import get_dataset, get_statistics


def get_metrics(config, pool=None):
    """
    code:from moses :
    """
    if config.gen_dir is None:
        raise KeyError
    gen = read_smiles_csv(os.path.join(config.gen_dir, "smiles.csv"))

    if config.metric_data_dir is None:
        test = get_dataset('test')
        ptest = get_statistics('test')
        test_scaffolds = get_dataset('test_scaffolds')
        ptest_scaffolds = get_statistics('test_scaffolds')
        train = get_dataset('train')
    else:
        test = read_smiles_csv(os.path.join(config.metric_data_dir, "test.csv"))
        ptest = np.load(
            os.path.join(config.metric_data_dir, "test_stats.npz"),
            allow_pickle=True)['stats'].item()
        test_scaffolds = read_smiles_csv(os.path.join(config.metric_data_dir, "test_scaffolds.csv.csv"))
        ptest_scaffolds = np.load(
            os.path.join(config.metric_data_dir, "test_scaffolds_stats.npz"),
            allow_pickle=True)['stats'].item()
        train = read_smiles_csv(os.path.join(config.metric_data_dir, "train.csv"))

    if config.unique_k is None:
        config.unique_k = [1000, 10000]
    disable_rdkit_log()
    metrics = {}
    close_pool = False

    if pool is None:
        if config.n_jobs != 1:
            pool = Pool(config.n_jobs)
            close_pool = True
        else:
            pool = 1
    metrics['valid'] = fraction_valid(gen, n_jobs=pool)
    gen = remove_invalid(gen, canonize=True)  ## 移除无效的分子
    if not isinstance(config.unique_k, (list, tuple)):
        config.unique_k = [config.unique_k]
    for _k in config.unique_k:
        metrics['unique@{}'.format(_k)] = fraction_unique(gen, _k, pool)  # 计算独特分子

    if ptest is None:
        ptest = compute_intermediate_statistics(test,
                                                n_jobs=config.n_jobs,
                                                device=config.device,
                                                batch_size=config.batch_size,
                                                pool=pool)

    if ptest_scaffolds is None:
        ptest_scaffolds = compute_intermediate_statistics(test_scaffolds,
                                                          n_jobs=config.n_jobs,
                                                          device=config.device,
                                                          batch_size=config.batch_size,
                                                          pool=pool)
    mols = mapper(pool)(get_mol, gen)
    kwargs = {'n_jobs': pool, 'device': config.device, 'batch_size': config.batch_size}
    kwargs_fcd = {'n_jobs': config.n_jobs, 'device': config.device, 'batch_size': config.batch_size}
    metrics['FCD/Test'] = FCDMetric(**kwargs_fcd)(gen=gen, pref=ptest['FCD'])  # Fréchet ChemNet 距离
    metrics['SNN/Test'] = SNNMetric(**kwargs)(gen=mols, pref=ptest['SNN'])  # 基分子指纹相似度
    metrics['Frag/Test'] = FragMetric(**kwargs)(gen=mols, pref=ptest['Frag'])  # 基于片段的相似度
    metrics['Scaf/Test'] = ScafMetric(**kwargs)(gen=mols, pref=ptest['Scaf'])  ##基于分子骨架相似度

    if ptest_scaffolds is not None:
        metrics['FCD/TestSF'] = FCDMetric(**kwargs_fcd)(
            gen=gen, pref=ptest_scaffolds['FCD']
        )
        metrics['SNN/TestSF'] = SNNMetric(**kwargs)(
            gen=mols, pref=ptest_scaffolds['SNN']
        )
        metrics['Frag/TestSF'] = FragMetric(**kwargs)(
            gen=mols, pref=ptest_scaffolds['Frag']
        )
        metrics['Scaf/TestSF'] = ScafMetric(**kwargs)(
            gen=mols, pref=ptest_scaffolds['Scaf']
        )

    metrics['IntDiv'] = internal_diversity(mols, pool,
                                           device=config.device)  # 1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    metrics['IntDiv2'] = internal_diversity(mols, pool, device=config.device, p=2)
    metrics['Filters'] = fraction_passes_filters(mols, pool)  # 检查 mol 是否通过 MCF 和 PAINS 过滤器，只允许原子不带电

    # Properties
    for name, func in [('logP', logP), ('SA', SA),
                       ('QED', QED),
                       ('weight', weight)]:
        metrics[name] = WassersteinMetric(func, **kwargs)(
            gen=mols, pref=ptest[name])

    if train is not None:
        metrics['Novelty'] = novelty(mols, train, pool)  # len(gen_smiles_set - train_set) /len(gen_smiles_se)
    enable_rdkit_log()
    if close_pool:
        pool.close()
        pool.join()
    return metrics


import evaluate

METRIC_PATH = os.path.dirname(__file__)


def valid(smiles: List[Text], n_jobs: int = 1):
    metric = evaluate.load(os.path.join(METRIC_PATH, "valid/valid.py"))
    return metric.compute(smiles=smiles, n_jobs=n_jobs)


def unique(predictions: List[Text]):
    pass


def logp(predictions: List[Text]):
    pass
