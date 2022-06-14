import numpy as np
import pandas as pd
from keras.models import load_model


def compute_total_gain_rate(day_gain):
    # 计算总收益率的函数
    # 其中day_gain是一个n*1的矩阵，表示每天收盘时的总资产量
    gain_rate = -1
    for i in range(1, day_gain.shape[0]):
        gain_rate += day_gain[i]/day_gain[i-1]
    return gain_rate*100


def compute_im_rate():
    # 计算信息比率
    pass


def compute_withdraw_rate():
    # 计算回撤率
    pass
