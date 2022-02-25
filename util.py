#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/8/18 20:54
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : util.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.0 - ZL.Z：2021/8/18
# 		      First version.
# @License  : None
# @Brief    : 通用方法集合

import csv


def write_csv(data, filename):
    """写入csv文件"""
    if not filename.endswith('.csv'):  # 补上后缀名
        filename += '.csv'
    # 中文需要设置成utf-8格式,为了防止excel打开中文乱码，使用utf-8-sig编码，会在文本前插入签名\ufeff，
    # 表示BOM(Byte Order Mark)，是一个不显示的标识字段，表示字节流高位在前还是低位在前
    with open(filename, "a", newline="", encoding="utf-8-sig") as f:  # 以追加模式、无新行、utf-8编码打开文件
        f_csv = csv.writer(f)  # 先写入表头
        # for item in data:
        #     f_csv.writerow(item)  # 一行一行写入
        f_csv.writerows(data)  # 写入多行


def read_csv(filename):
    """读取csv文件"通过csv.reader()来打开csv文件，返回的是一个列表格式的迭代器，
    可以通过next()方法获取其中的元素，也可以使用for循环依次取出所有元素。"""
    if not filename.endswith(".csv"):  # 补上后缀名
        filename += ".csv"
    data = []  # 所读到的文件数据
    with open(filename, "r", encoding="utf-8-sig") as f:  # 以读模式、utf-8编码打开文件
        f_csv = csv.reader(f)  # f_csv对象，是一个列表的格式
        for row in f_csv:
            data.append(row)
    return data


def vup_duration_from_vuvInfo(vuv_info: str, p_dur_thr=140):
    """
    从praat的vuv.TextGrid对象中List得到的Info文本信息分割语音：浊音段voice segments、清音段unvoice segments、停顿段(静音段)pause segments
    :param vuv_info: vuv.TextGrid对象中List得到的Info文本信息
    :param p_dur_thr: 停顿段时长阈值，单位ms，大于该阈值的清音段归类为停顿段，默认为140ms
    :return: segments_voice, segments_unvoice, segments_pause,
            float, list(n_segments, 2),对应每一段的起始和结束时间，单位为s
    """
    segments_voice, segments_unvoice, segments_pause = [], [], []
    for text_line in vuv_info.strip('\n').split('\n'):
        text_line = text_line.split('\t')
        if 'text' not in text_line:
            tmin = float(text_line[0])
            text = text_line[1]
            tmax = float(text_line[2])
            if text == 'V':  # voice段
                segments_voice.append([tmin, tmax])
            elif text == 'U':  # unvoice段,其中包含pause段
                duration = 1000 * (tmax - tmin)
                if duration > p_dur_thr:  # 若间隔时间超过设置的阈值，则判定该段为pause段
                    segments_pause.append([tmin, tmax])
                else:
                    segments_unvoice.append([tmin, tmax])
    return segments_voice, segments_unvoice, segments_pause


def vup_duration_from_vuvTextGrid(vuv_file, p_dur_thr=140):
    """
    从praat得到的vuv.TextGrid文件中分割语音：浊音段voice segments、清音段unvoice segments、停顿段(静音段)pause segments
    :param vuv_file: 从praat得到的vuv.TextGrid文件
    :param p_dur_thr: 停顿段时长阈值，单位ms，大于该阈值的清音段归类为停顿段，默认为140ms
    :return: segments_voice, segments_unvoice, segments_pause,
            float, list(n_segments, 2),对应每一段的起始和结束时间，单位为s
    """
    with open(vuv_file) as f:
        segments_voice, segments_unvoice, segments_pause = [], [], []
        data_list = f.readlines()
        for data_index in range(len(data_list)):
            data_list[data_index] = data_list[data_index].strip()  # 去掉换行符
            if data_list[data_index] == '"V"':  # voice段
                # 标识字符前两行为起始的duration
                segments_voice.append([float(data_list[data_index - 2]), float(data_list[data_index - 1])])
            elif data_list[data_index] == '"U"':  # unvoice段,其中包含pause段
                duration = 1000 * (float(data_list[data_index - 1]) - float(data_list[data_index - 2]))
                if duration > p_dur_thr:  # 若间隔时间超过设置的阈值，则判定该段为pause段
                    segments_pause.append([float(data_list[data_index - 2]), float(data_list[data_index - 1])])
                else:
                    segments_unvoice.append([float(data_list[data_index - 2]), float(data_list[data_index - 1])])
        return segments_voice, segments_unvoice, segments_pause





