import backtrader.feeds as btfeeds
import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

file = r"D:\vs_code_files\python\projects\python程序\数学建模\mathematical_modeling\a.xlsx"
predict = "收盘价"


def compute_total_gain_rate(day_gain):
    # 计算总收益率的函数
    # 其中day_gain是一个n*1的矩阵，表示每天收盘时的总资产量
    # day_gain[0]是第0天总收益，也就是投资资本
    gain_rate = 1
    for i in range(1, day_gain.shape[0]):
        gain_rate *= ((day_gain[i]-day_gain[i-1])/day_gain[i-1]+1)
    return (gain_rate-1)*100


def compute_im_rate(day_gain, zzzs):
    # 计算信息比率
    # day_gain是每天交易后的资产总数
    # zzzs是中正指数

    # 先处理成np矩阵形式
    zzzs = np.array(zzzs)
    zzzs = zzzs.reshape((-1, 1))
    day_gain = np.array(day_gain)
    day_gain = day_gain.reshape((-1, 1))


def compute_withdraw_rate():
    # 计算回撤率
    pass


def get_data():
    with open(file, 'rb') as f:
        szjj = pd.read_excel(f, sheet_name=3)
        szjj = szjj.iloc[:-2, :]
        szjj = szjj.iloc[912:, :]
        szjj = szjj.iloc[::-1, :]
        # 获取到每天的经济指标数据

        gnsc = pd.read_excel(f, sheet_name=2)
        gnsc = gnsc.drop(index=[0, 1])
        gnsc = gnsc[["中证500指数"]]
        gnsc = gnsc.iloc[:19, :]
        zzzs = gnsc.iloc[::-1, :]
        # 此时获取到待预测的每一天中的中正500指数和每天5分钟的股票数据
        return szjj, zzzs


class my_strategy1(bt.Strategy):
    # 全局设定交易策略的参数
    params = (
        ('maperiod', 20),
    )

    def __init__(self):
        # 指定价格序列
        self.dataclose = self.datas[0].close
        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # 添加移动均线指标，内置了talib模块
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    def next(self):
        if self.order:  # 检查是否有指令等待执行,
            return
        # 检查是否持仓
        if not self.position:  # 没有持仓
            # 执行买入条件判断：收盘价格上涨突破20日均线
            if self.dataclose[0] > self.sma[0]:
                # 执行买入
                self.order = self.buy(size=50)
        else:
            # 执行卖出条件判断：收盘价格跌破20日均线
            if self.dataclose[0] < self.sma[0]:
                # 执行卖出
                self.order = self.sell(size=100)
