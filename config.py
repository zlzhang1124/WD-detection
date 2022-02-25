#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/8/26
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : config.py
# @Software : Python3.6;PyCharm;Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.0
# @License  : None
# @Brief    : 配置文件

import os
import sys
import matplotlib
import warnings
import platform
import pandas as pd
import random
import numpy as np
# import tensorflow as tf
# import torch
# from torch.backends import cudnn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # 设置tensorflow输出控制台信息：1等级，屏蔽INFO，只显示WARNING + ERROR + FATAL
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 5)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 200)
np.set_printoptions(threshold=np.inf)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'


if platform.system() == 'Windows':
	DATA_PATH = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\安中医神经病学研究所合作\data\preprocessed_data"
	DATA_PATH_EXT = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\言语特征可重复性\data\preprocessed_data"
	# font_family = 'Times New Roman'
	font_family = 'Arial'
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第0个GPU,-1不使用GPU，使用CPU
else:
	DATA_PATH = r"/home/zlzhang/data/WD_PD/data/preprocessed_data"
	DATA_PATH_EXT = r"/home/zlzhang/data/言语特征可重复性/data/preprocessed_data"
	font_family = 'DejaVu Sans'
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 使用第0-2个GPU,-1不使用GPU，使用CPU
	# gpus = tf.config.list_physical_devices('GPU')
	# tf.config.experimental.set_memory_growth(gpus[0], True)  # 防止GPU显存爆掉
matplotlib.rcParams["font.family"] = font_family


def setup_seed(seed: int):
	"""
	全局固定随机种子
	:param seed: 随机种子值
	:return: None
	"""
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	# os.environ['TF_DETERMINISTIC_OPS'] = '1'
	# tf.random.set_seed(seed)
	# torch.manual_seed(seed)
	# if torch.cuda.is_available():
	# 	torch.cuda.manual_seed(seed)
	# 	torch.cuda.manual_seed_all(seed)
	# 	cudnn.deterministic = True
	# 	cudnn.benchmark = False
	# 	cudnn.enabled = False


rs = 323
setup_seed(rs)


