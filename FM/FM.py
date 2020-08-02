from __future__ import division
from math import exp
from numpy import *
from random import normalvariate  # 正态分布
import time
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split



def data_preprocessing(input_data):
    """预处理数据
    :param input_data: 输入的数据
    :return 数据的特征feature, 数据的标签label
    """
    feature = np.array(input_data.iloc[:, :-1])  # 取特征(8个特征)
    label = input_data.iloc[:, -1].map(lambda x: 1 if x == 1 else -1)  # 取Outcome标签并转化为 +1，-1

    # 将数组按行进行归一化
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)  # 特征的最大值，特征的最小值
    feature = (feature - zmin) / (zmax - zmin)
    label = np.array(label)

    return feature, label

def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))

def FM(dataMatrix, classLabels, k, iter):
    '''
    :param dataMatrix:  特征矩阵
    :param classLabels: 类别矩阵
    :param k:           隐向量特征维度
    :param iter:        迭代次数
    :return:            常数项w_0, 一阶特征系数w, 二阶交叉特征系数v
    '''
    # dataMatrix用的是matrix, classLabels是列表
    m, n = shape(dataMatrix)  # 矩阵的行列数，即样本数m和特征数n
    alpha = 0.03

    # 初始化参数
    w_0 = 0  # 常数项初始系数
    w_1 = zeros((n, 1))  # 一阶特征的初始系数
    v = normalvariate(0, 0.2) * ones((n, k))  # 二阶交叉特征的系数初始系数

    for it in range(iter):
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v  # 每个样本(1*n)x(n*k),得到k维向量
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # 二阶交叉项计算，得到k维向量
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.  # 二阶交叉项计算完成

            p = w_0 + dataMatrix[x] * w_1 + interaction  # 计算预测的输出，即FM的全部项之和
            loss = sigmoid(classLabels[x] * p[0, 0]) -1  # 损失函数公共项目
            w_0 = w_0 - alpha * loss * classLabels[x]  #更新参数项参数

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w_1[i, 0] = w_1[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i] #更新一阶项参数
                    for j in range(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (
                                dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i]) #更新而二阶项目参数
        #print("第{}次迭代后的损失为{}".format(it, loss))
        print("\t------- iter: ", it, " , cost: ", \
        getCost(getPrediction(np.mat(dataTrain), w_0, w_1, v), classLabels))
    # 常数项w_0, 一阶特征系数w_1（n维向量——n个特征）, 二阶交叉特征系数v（n个k维向量）
    return w_0, w_1, v


def getCost(predict, classLabels):
    '''计算预测准确性
    input:  predict(list)预测值
            classLabels(list)标签
    output: error(float)计算损失函数的值
    '''
    m = len(predict)
    error = 0.0
    for i in range(m):
        error -= np.log(sigmoid(predict[i] * classLabels[i]))
    return error


def getPrediction(dataMatrix, w_0, w_1, v):
    '''得到预测值
    input:  dataMatrix(mat)特征
            w(int)常数项权重
            w0(int)一次项权重
            v(float)交叉项权重
    output: result(list)预测的结果
    '''
    m = np.shape(dataMatrix)[0]
    result = []
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
                  np.multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w_1 + interaction  # 计算预测的输出
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result

def getAccuracy(predict, classLabels):
    '''计算预测准确性
    input:  predict(list)预测值
            classLabels(list)标签
    output: float(error) / allItem(float)错误率
    '''
    m = len(predict)
    allItem = 0
    error = 0
    for i in range(m):
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue
    return float(error) / allItem

def save_model(file_name, w0, w, v):
    '''保存训练好的FM模型
    input:  file_name(string):保存的文件名
            w0(float):偏置项
            w(mat):一次项的权重
            v(mat):交叉项的权重
    '''
    f = open(file_name, "w")
    # 1、保存w0
    f.write(str(w0) + "\n")
    # 2、保存一次项的权重
    w_array = []
    m = np.shape(w)[0]
    for i in range(m):
        w_array.append(str(w[i, 0]))
    f.write("\t".join(w_array) + "\n")
    # 3、保存交叉项的权重
    m1 , n1 = np.shape(v)
    for i in range(m1):
        v_tmp = []
        for j in range(n1):
            v_tmp.append(str(v[i, j]))
        f.write("\t".join(v_tmp) + "\n")
    f.close()

if __name__ == '__main__':
    diabetes_file = './data/diabetes.csv'
    input_data = pd.read_csv(diabetes_file)
    train, test = train_test_split(input_data, random_state=1)
    dataTrain, labelTrain = data_preprocessing(train)
    dataTest, labelTest = data_preprocessing(test)
    date_startTrain = time.time()
    print("开始训练")
    w_0, w_1, v = FM(mat(dataTrain), labelTrain, 20, 1000)
    predict_train_result = getPrediction(np.mat(dataTrain), w_0, w_1, v)
    print("训练准确性为：%f" % (1 - getAccuracy(predict_train_result, labelTrain)))
    date_endTrain = time.time()
    print("训练用时为：%.2f" % (date_endTrain - date_startTrain), "s")

    print("开始测试")
    predict_test_result = getPrediction(np.mat(dataTest), w_0, w_1, v)
    print("测试准确性为：%f" % (1 - getAccuracy(predict_test_result , labelTest)))
    # 3、保存训练好的FM模型
    save_model("data/weights", w_0, w_1, v)