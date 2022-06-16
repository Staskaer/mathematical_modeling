import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from functools import reduce
from copy import deepcopy
import pickle

file = r"D:\vs_code_files\python\projects\python程序\数学建模\mathematical_modeling\a.xlsx"
# 基于皮尔逊相关系数
best = ['VMA', 'VMACD', '成交金额:上证综合指数', '互联网电商', '创业板指数', '沪深300指数', 'EXPMA',
        '成交量:上证综合指数', 'MA', 'BBI', '深证成份指数', '恒生指数', '俄罗斯RTS指数', 'BIAS', 'BOLL']
y = "成交量"
lookback = 50
max_data = 884910620
min_data = 0


def get_data(type_data=False, day=True):
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
        if day:
            # 只提取每一天的数据
            szjj = szjj[szjj["时间"].map(
                lambda x: str(x).split(" ")[1] == "15:00:00")]
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
    test = dataset.iloc[-912:, :]
    train = dataset.iloc[:-912, :]
    # train = dataset.iloc[0:-19, :]
    # test = dataset.iloc[-19:, :]

    trainX = train[best[0:10]]
    trainY = train[y]
    testX = test[best[0:10]]
    testY = test[y]

    return trainX, trainY, testX, testY


def train_and_save(dataset):

    trainX, trainY, testX, testY = get_train_test(dataset)
    X_all = pd.concat((trainX, testX), axis=0)
    Y_all = pd.concat((trainY, testY), axis=0)
    X_all = pd.concat((X_all, Y_all), axis=1)
    X_all, Y_all = np.array(X_all), np.array(Y_all)

    scaler = MinMaxScaler()
    trainX = scaler.fit_transform(X_all)
    trainY = scaler.fit_transform(np.array(Y_all).reshape(-1, 1))

    x_train, y_train = [], []
    for i in range(lookback, trainX.shape[0]):
        x_train.append(trainX[i-lookback:i, :])
        y_train.append(trainY[i])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], lookback, 11))
    y_train = y_train.reshape(y_train.shape[0], 1)

    # 对矩阵reshape成[samples, time steps, features]
    # trainX = np.array(trainX)
    # trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))

    # LSTM网络模型
    # 3个LSTM层，4个全连接层，拟合能力拉满
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True))
    # model.add(LSTM(units=100, return_sequences=True))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(Dense(units=64))
    # model.add(Dense(units=32))
    # model.add(Dense(units=16))
    # model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mae')
    history = model.fit(x_train, y_train, epochs=250, batch_size=32, verbose=2)
    model.save(
        r"D:\vs_code_files\python\projects\python程序\数学建模\mathematical_modeling\q2\w.h5")
    # model.save(
    #     r"D:\vs_code_files\python\projects\python程序\数学建模\mathematical_modeling\q2\model.h5")

    # plt.title("训练损失图")
    # plt.plot(history.history['loss'])
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.show()
    with open(r"D:\vs_code_files\python\projects\python程序\数学建模\mathematical_modeling\q2\history", "wb") as f:
        pickle.dump(history.history, f)


def test(dataset):
    model = load_model(
        r"D:\vs_code_files\python\projects\python程序\数学建模\mathematical_modeling\q2\w.h5")
    # 使用训练好的模型进行预测，因为训练模型需要很长时间
    # model = load_model(
    #     r"D:\vs_code_files\python\projects\python程序\数学建模\mathematical_modeling\q2\model.h5")
    trainX, trainY, testX, testY = get_train_test(dataset)
    # 上面这一堆是对数据进行处理，忘了封装了，只能这样子单独处理了
    X_all = pd.concat((trainX, testX), axis=0)
    Y_all = pd.concat((trainY, testY), axis=0)
    X_all, Y_all = np.array(X_all), np.array(Y_all)
    # X_all = X_all[len(X_all) - len(testY) - 1:]
    # Y_all = Y_all[len(X_all) - len(testY) - 1:]
    x_test, y_test = [], []
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(X_all)
    # testX = scaler.fit_transform(testX)
    # trainX = scaler.fit_transform(trainX)

    # testX = np.array(testX)
    # testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
    # trainX = np.array(trainX)
    # trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    for i in range(lookback, X_all.shape[0]):
        x_test.append(X_all[i-lookback:i, :])
        y_test.append(Y_all[i])
    x_test, y_test = np.array(x_test), np.array(y_test)
    # x_test = x_test.reshape((x_test.shape[0], 5, 10))

    # 预测
    trainPredict = model.predict(x_test)
    real_price = scaler.fit_transform(np.array(y_test).reshape(-1, 1))

    trainPredict = np.array(trainPredict)
    real_price = np.array(real_price)

    trainPredict = trainPredict*(max_data-min_data) + min_data
    real_price = real_price*(max_data-min_data) + min_data

    with open(r"D:\vs_code_files\python\projects\python程序\数学建模\mathematical_modeling\q2\history", "rb") as f:
        history = pickle.load(f)

    plt.subplot(221)
    plt.plot(real_price, color="g", label="raw")
    plt.plot(trainPredict, color="r", label="predicted")
    plt.legend(["raw", "predicted"])
    plt.title("预测与真实数据对比")
    plt.xlabel("时间")

    plt.subplot(223)
    plt.title("训练损失图")
    plt.plot(history['loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(222)
    plt.plot(real_price, color="g", label="raw")
    plt.legend(["raw"])
    plt.title("真实数据")
    plt.xlabel("时间")

    plt.subplot(224)
    plt.plot(trainPredict, color="r", label="predicted")
    plt.legend(["predicted"])
    plt.title("预测值")
    plt.xlabel("时间")

    plt.show()


if __name__ == "__main__":

    # 获取数据然后归一化
    dataset, _ = get_data(day=False)
    dataset = dataset.drop(columns="时间")
    dataset = dataset.astype("float32")

    train_and_save(dataset)  # 训练模型并验证
    test(dataset)
    # 不知道怎么回事，模型保存再打开就会报错

    # 获取数据集和验证集
