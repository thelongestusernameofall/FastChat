#!/usr/bin/env python
# coding=utf-8

## 该脚本当前在IntelliSolver项目中实现， 文件名：extra/Train/rlhf/clean-rlhf-samples.py

"""
    根据SFT的语料，以及模型当前的生成结果，构造强化学习需要的语料
    sft: prompt, sft-answer
    首先收集: prompt, sft-answer, model-answer
    然后根据sft-answer和model-answer的相似度或者训练一个奖励模型来生成评分，构造强化学习的语料
    rl: prompt, model-answer, reward
"""
import argparse
import json
import os
import random
import sys
import time


def get_model_answer(prompts: list):
    model_answers = []
    return model_answers
