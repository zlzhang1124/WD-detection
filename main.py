#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/2/4 16:32
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : main.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.0: 2022/2/4
#             First version.
# @License  : None
# @Brief    : 基于信号分解的MFCC特征，用于WD分类


if __name__ == "__main__":
    pass
    # Step1: audio_preprocess.py, 音频与处理，获取所需的音频数据
    # Step2: signal_decomposition.py, 语音信号分解，分别利用EMD/CEEMDAN/VMD
    # Step3: calcu_features.py, 基于分解的信号进行特征计算，包括MFCC/HCC/DMFCC（首先确定每种分解方法对应的使用IMF数量）
    # Step4: models.py, 建模
