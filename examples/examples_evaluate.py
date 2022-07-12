#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/11 23:57
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : examples_evaluate.py
# import evaluate
# valid = evaluate.load("./valid/valid.py")
# score = valid.compute(predictions=["CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1","CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1"], n_jobs=1)
# valid = evaluate.load("./accuracy/accuracy.py")
# score = valid.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
# print(score)
from drugai.metrics.metric_util import valid

score = valid(smiles=["CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1","CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1"], n_jobs=1)
print(score)
