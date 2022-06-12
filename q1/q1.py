
# 依赖项：pandas numpy matplotlib sklearn minepy

from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from scipy.stats import pearsonr
from minepy import MINE

file = r"D:\vs_code_files\python\projects\python程序\数学建模\mathematical_modeling\a.xlsx"


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
        szjj = szjj.iloc[::-1, :]
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
    plt.subplot(231)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 4], label="收盘价", color="r")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("收盘价随时间的变化走势图")
    plt.xticks([])

    plt.subplot(232)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 3], label="开盘价", color="g")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("收盘价随时间的变化走势图")
    plt.xticks([])

    plt.subplot(234)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 5], label="最高价", color="b")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("最高价随时间的变化走势图")
    plt.xticks([])

    plt.subplot(235)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 6], label="最低价", color="y")
    plt.xlabel("时间")
    plt.ylabel("价格")
    plt.title("最低价随时间的变化走势图")
    plt.xticks([])

    plt.subplot(233)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 7], label="成交量", color="c")
    plt.xlabel("时间")
    plt.ylabel("成交量")
    plt.title("成交量随时间的变化走势图")
    plt.xticks([])

    plt.subplot(236)
    plt.plot(szjj.iloc[:, 0], szjj.iloc[:, 8], label="成交额", color="m")
    plt.xlabel("时间")
    plt.ylabel("成交额")
    plt.title("成交额随时间的变化走势图")
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


def feature_select(X, y):
    # 特征提取方法
    result = {}
    '''（一）卡方(Chi2)检验'''
    # 先去除nan
    X = X.fillna(0)
    # raw是原始数据，用于记录列号和名字
    raw = deepcopy(X)

    # 归一化处理
    scalar = MinMaxScaler()
    X = scalar.fit_transform(X)

    # 此处是调用卡方检测来挑选出最优的15个特征
    selector = SelectKBest(chi2, k=15)
    X_new = selector.fit_transform(X, y.astype('int'))

    # 下面的部分是显示这些特征对应的类别
    scores = selector.scores_
    indices = np.argsort(scores)[::-1]
    k_best_list_k = []
    for i in range(15):
        k_best_feature = raw.columns[indices[i]]
        result[k_best_feature] = result.get(k_best_feature, 0)+15-i
        k_best_list_k.append(k_best_feature)
    print('使用卡方检验得到的最优的15个特征：', k_best_list_k)

    '''（二）皮尔逊'''
    # 皮尔逊（这部分我看知乎里有提到，卡方对分类，皮尔逊对回归）
    # 而这个正是一个回归问题，所以把这个也用上

    selector = SelectKBest(lambda X, Y: np.array(
        list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=15)
    X_new = selector.fit_transform(X, y.astype('int'))

    # 下面的部分是显示这些特征对应的类别
    scores = selector.scores_
    indices = np.argsort(scores)[::-1]
    k_best_list_p = []
    for i in range(15):
        k_best_feature = raw.columns[indices[i]]
        result[k_best_feature] = result.get(k_best_feature, 0)+15-i
        k_best_list_p.append(k_best_feature)
    print('使用皮尔逊检验得到的最优的15个特征：', k_best_list_p)

    '''（三）基于互信息'''
    def mic(x, y):
        # 计算函数
        m = MINE()
        m.compute_score(x, y)
        return (m.mic(), 0.5)

    selector = SelectKBest(lambda X, Y: np.array(
        list(map(lambda x: mic(x, Y), X.T))).T[0], k=15)
    X_new = selector.fit_transform(X, y.astype('int'))

    # 下面的部分是显示这些特征对应的类别
    scores = selector.scores_
    indices = np.argsort(scores)[::-1]
    k_best_list_h = []
    for i in range(15):
        k_best_feature = raw.columns[indices[i]]
        result[k_best_feature] = result.get(k_best_feature, 0)+15-i
        k_best_list_h.append(k_best_feature)
    print('使用互信息得到的最优的15个特征：', k_best_list_h)

    print("========最终投票结果（前6）==========")
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    print(result[0:6])
    #这个排序其实写的不对，后面再改改看看#


def parse_analysis(data):
    print("分析哪些特征跟开盘价最相关:\n")
    X, y = get_X_y(data=data, y_type=0)
    feature_select(X, y)

    print("\n分析哪些特征跟收盘价最相关:\n")
    X, y = get_X_y(data=data, y_type=1)
    feature_select(X, y)

    print("\n分析哪些特征跟最高价最相关:\n")
    X, y = get_X_y(data=data, y_type=2)
    feature_select(X, y)

    print("\n分析哪些特征跟最低价最相关:\n")
    X, y = get_X_y(data=data, y_type=3)
    feature_select(X, y)

    print("\n分析哪些特征跟交易量最相关:\n")
    X, y = get_X_y(data=data, y_type=4)
    feature_select(X, y)

    print("\n分析哪些特征跟交易价最相关:\n")
    X, y = get_X_y(data=data, y_type=5)
    feature_select(X, y)


if __name__ == "__main__":
    data, szjj = get_data()  # 获取数据
    # print(data.columns)

    # parse_analysis(data)  # 这个用于分析相关性
    show(szjj)  # 这个用于绘图
