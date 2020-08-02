import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data  = pd.read_csv('./data/Advertising.csv')
data = data.drop(['index'] ,axis=1)
data = (data - data.mean()) / data.std()

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
#为了使得计算方便，新增一特征ones
data.insert(0 ,'ones' ,1)
X = data[['ones' ,'TV','radio','newspaper']]
y = data[['sales']]


#划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
y_test_1 = y_test
X = X_train
y = y_train
#将输入值转为Numpy矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)

X_test = np.matrix(X_test.values)
y_test = np.matrix(y_test.values)
#初始化权重稀系数
theta = np.matrix(np.array([0,0,0,0] ))

# 批量梯度下降
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters  = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error , X[:,j])
            temp[0,j] = theta[0,j] - alpha * np.sum(term) / len(X)
        theta = temp
        cost[i] = computeCost(X,y,theta)

    return theta ,cost

iners = 1000
g, cost = gradientDescent(X, y, theta, 0.001, iners)

print(computeCost(X_test, y_test, g))

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iners), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
