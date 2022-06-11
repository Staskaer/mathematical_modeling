
# 依赖项：pandas numpy matplotlib sklearn

from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from functools import reduce

file = r"projects\python程序\数学建模\a.xlsx"

'''
#这部分代码弃用

def get_szjj():
    # 打开数字经济板块信息并提取出所有的数据
    with open(file, 'rb') as f:
        df = pd.read_excel(f, sheet_name=3)
        # 除去最后两行不必要的信息
        df = df.iloc[:-2, :]
        return df


def get_jszb():
    # 打开技术指标板块并提取出所有的数据
    with open(file, 'rb') as f:
        df = pd.read_excel(f, sheet_name=4)
        return df


def get_gjsc():
    # 获取国际市场数据
    with open(file, 'rb') as f:
        df = pd.read_excel(f, sheet_name=5)
        return df


def get_hl():
    # 获取汇率相关数据
    with open(file, 'rb') as f:
        df = pd.read_excel(f, sheet_name=6)
        return df


def get_qt():
    # 获取其他板块的信息
    with open(file, 'rb') as f:
        df = pd.read_excel(f, sheet_name=7)
        return df


def match(szjj, jszb):
    # 因为数字经济板块和技术指标中的时间不匹配
    # 并且技术指标中数据明显少很多，因此将提取出匹配的数据用于处理
    szjj['时间'] = szjj['时间'].map(lambda x: str(x).split(" ")[0])
    jszb['时间'] = jszb['时间'].map(lambda x: str(x).split(" ")[0])
    # 使用内连接的方式来获取将其组装成整体
    result = pd.merge(szjj, jszb, on="时间")

'''


def get_data(type_data=False):
    # 获取拼接后的信息
    with open(file, 'rb') as f:
        # 宏观指标1
        hgzb1 = pd.read_excel(f, sheet_name=0)
        hgzb1 = hgzb1.drop(index=[0, 1])
        hgzb1 = hgzb1.rename(columns={"指标名称": "时间"})
        hgzb1['时间'] = hgzb1['时间'].map(lambda x: str(x).split(" ")[0])
        # 宏观指标2
        hgzb2 = pd.read_excel(f, sheet_name=1)
        hgzb2 = hgzb2.drop(index=[0, 1])
        hgzb2 = hgzb2.rename(columns={"指标名称": "时间"})
        hgzb2['时间'] = hgzb2['时间'].map(lambda x: str(x).split(" ")[0])
        # 国内市场指标
        gnsc = pd.read_excel(f, sheet_name=2)
        gnsc = gnsc.drop(index=[0, 1])
        gnsc = gnsc.rename(columns={"指标名称": "时间"})
        gnsc['时间'] = gnsc['时间'].map(lambda x: str(x).split(" ")[0])
        # 数字经济板块
        szjj = pd.read_excel(f, sheet_name=3)
        szjj = szjj.iloc[:-2, :]
        szjj['时间'] = szjj['时间'].map(lambda x: str(x).split(" ")[0])
        szjj_raw = deepcopy(szjj)  # 备份绘图用
        # 技术指标
        jszb = pd.read_excel(f, sheet_name=4)
        jszb['时间'] = jszb['时间'].map(lambda x: str(x).split(" ")[0])
        # 国际市场指标
        gjsc = pd.read_excel(f, sheet_name=5)
        gjsc = gjsc.drop(index=[0, 1])
        gjsc = gjsc.rename(columns={"指标名称": "时间"})
        gjsc['时间'] = gjsc['时间'].map(lambda x: str(x).split(" ")[0])
        # 汇率
        hl = pd.read_excel(f, sheet_name=6)
        hl = hl.drop(index=[0, 1])
        hl = hl.rename(columns={"指标名称": "时间"})
        hl['时间'] = hl['时间'].map(lambda x: str(x).split(" ")[0])
        # 其他信息
        qt = pd.read_excel(f, sheet_name=7)
        qt = qt.iloc[:-2, :]
        qt = qt.rename(columns={"日期": "时间"})
        qt['时间'] = qt['时间'].map(lambda x: str(x).split(" ")[0])

        # 合并
        if type_data:
            dfs = [szjj, hgzb1, hgzb2, gnsc,  jszb, gjsc, hl, qt]
        # 宏观指标跨度太大，对实时性分析的数据意义感觉不大
        else:
            dfs = [szjj, gnsc,  jszb, gjsc, hl, qt]
        result = reduce(lambda left, right: pd.merge(
            left, right, how="left", on='时间'), dfs)
        # 将数字经济中的时间作为主键，用左连接的方式来整合数据
        result = result.drop(columns=["序号_x", "序号_y", "指数代码_x", "指数代码_y"])
        return result, szjj_raw


def show(szjj):
    # 绘制走势图
    # 数据太大，绘图太慢
    plt.subplot(221)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 4], label="收盘价", color="r")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("收盘价随时间的变化走势图")
    plt.xticks([])

    plt.subplot(222)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 3], label="开盘价", color="g")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("收盘价随时间的变化走势图")
    plt.xticks([])

    plt.subplot(223)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 5], label="最高价", color="b")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("最高价随时间的变化走势图")
    plt.xticks([])

    plt.subplot(224)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 6], label="最低价", color="y")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("最低价随时间的变化走势图")
    plt.xticks([])
    plt.show()

    # plt.savefig(r"projects\python程序\数学建模\1.png")
    # plt.savefig(r"projects\python程序\数学建模\1.svg")


def get_X_y(data, y_type=0):
    # 从处理好的数据中提取需要用到的数据
    # 其中y_type可以为0-5分别对应开盘 收盘 最高 最低 成交量 成交额
    # 个人理解有用的应该是收盘和成交额？

    X = data.iloc[:, 7:]
    y = data.iloc[:, y_type+1]
    return X, y


def compute_phi():

    # 计算相关系数
    pass


def compute_gain():
    # 信息增益或信息熵
    pass


def feature_select(X, y):
    # 特征提取方法
    '''（一）卡方(Chi2)检验'''
    # 先去除nan
    X = X.fillna(0)
    raw = deepcopy(X)

    # 归一化处理
    scalar = MinMaxScaler()
    X = scalar.fit_transform(X)

    selector = SelectKBest(chi2, k=10)
    X_new = selector.fit_transform(X, y.astype('int'))

    scores = selector.scores_

    indices = np.argsort(scores)[::-1]
    k_best_list = []

    for i in range(10):
        k_best_feature = raw.columns[indices[i]]
        k_best_list.append(k_best_feature)
    print('k_best_list:', k_best_list)


if __name__ == "__main__":
    data, szjj = get_data()
    X, y = get_X_y(data=data)
    feature_select(X, y)

    # print(data.columns.values)

    # show(szjj)  # 绘图比较慢

    # 先获取并清洗数据，然后有以下的步骤：
    # 1. 来计算相关系数
    # 2. 使用特征提取方法
    # 3. 使用降维技术
    pass
