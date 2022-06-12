import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from functools import reduce
from copy import deepcopy

file = r"projects\python程序\数学建模\mathematical_modeling\a.xlsx"
best = ['VMA', 'VMACD', '互联网电商', '成交金额:上证综合指数', '创业板指数', 'EXPMA', 'BBI',
        'MA', '沪深300指数', '成交量:上证综合指数', 'BOLL', 'OBV', '恒生指数', '俄罗斯RTS指数', '深证成份指数']
y = "成交量"


def get_data(type_data=False):
    # 获取拼接后的信息，复用第一问中提取数据的代码
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


def get_train_test(dataset):
    # 从数据集中获取训练集和交叉验证集
    test = dataset.iloc[0:912, :]
    train = dataset.iloc[912:, :]

    trainX = train[best[0:10]]
    trainY = train[y]
    testX = test[best[0:10]]
    testY = test[y]

    return trainX, trainY, testX, testY


def train_and_save(dataset):

    trainX, trainY, testX, testY = get_train_test(dataset)
    scaler = MinMaxScaler(feature_range=(0, 1))
    trainX = scaler.fit_transform(trainX)
    # trainY = scaler.fit_transform(trainY)
    # testX = scaler.fit_transform(testX)
    # testY = scaler.fit_transform(testY)
    # 对矩阵reshape成[samples, time steps, features]
    trainX = np.array(trainX)
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))

    # LSTM网络模型
    model = Sequential()
    model.add(LSTM(4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    model.save(r"projects\python程序\数学建模\mathematical_modeling\q2\model.h5")


def test(dataset):
    # 使用训练好的模型进行预测，因为训练模型需要很长时间
    model = load_model(
        r"projects\python程序\数学建模\mathematical_modeling\q2\model.h5")
    trainX, trainY, testX, testY = get_train_test(dataset)

    scaler = MinMaxScaler(feature_range=(0, 1))
    testX = scaler.fit_transform(testX)
    trainX = scaler.fit_transform(trainX)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    testX = np.array(testX)
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
    trainX = np.array(trainX)
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE'.format(trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE'.format(testScore))
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(1*2) +
                    1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


if __name__ == "__main__":

    # 获取数据然后归一化
    dataset, _ = get_data()
    dataset = dataset.drop(columns="时间")
    dataset = dataset.astype("float32")

    train_and_save(dataset)  # 训练模型并保存

    # 获取数据集和验证集
