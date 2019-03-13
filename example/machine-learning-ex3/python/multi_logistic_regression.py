# !/usr/bin/env python 
# -*- coding:utf-8 -*-
# 拟合逻辑回归模型，解决多分类问题
# 使用正则化，使用高级优化算法
# 作者：胡森
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.io as scio
import math


def display_data(x):
    """
    :param x: 输入的照片
    :return: 
    """
    # 设置单个图片展示宽度及高度
    width = round(math.sqrt(x.shape[1]))
    m, n = x.shape
    height = round(n / width)
    # 设置横向、纵向展示个数
    raw_num = math.floor(math.sqrt(x.shape[0]))
    col_num = math.ceil(m / raw_num)
    # 设置图片间距
    pad = 1
    # 展示数组
    display_array = - np.ones((pad + raw_num * (height + pad),
                               pad + col_num * (width + pad)));
    # 计数器
    count = 0
    for i in range(raw_num):
        for j in range(col_num):
            if count > m - 1:
                break
            max_val = max(x[count, :])
            display_array[pad + j * (width + pad):(j + 1) * (pad + width),
            pad + i * (height + pad):(i + 1) * (pad + height)] = np.reshape(x[count, :], (width, height)) / max_val
            count += 1
        if count > m:
            break
    plt.imshow(display_array.T, cmap=plt.cm.gray_r)
    plt.show()


# sigmoid函数（logistic函数）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, x, y, lamb):
    """
    :param theta: 权值
    :param x: 输入变量
    :param y: 输出变量
    :param lamb: 正则化系数
    :return: 代价函数
    """
    # 样本数
    m, n = x.shape
    theta = theta.reshape((-1, 1))

    cost = -1 / m * np.sum(np.log(sigmoid(x @ theta)) * y + (1 - y) * np.log(1 - sigmoid(x @ theta))) + lamb / (
        2 * m) * (np.sum(theta * theta) - theta[0] * theta[0])
    return cost


def grad(theta, x, y, lamb):
    """
    :param theta: 权值
    :param x: 输入变量
    :param y: 输出变量
    :param lamb: 正则化系数
    :return: 优化算子
    """
    # 样本数
    m, n = x.shape
    theta = theta.reshape((-1, 1))
    grad = np.zeros((n, 1))
    grad[0] = 1 / m * (sigmoid(x @ theta) - y).T @ x[:, 0]
    # for i in range(1, n):
    # grad[i] = 1 / m * (sigmoid(x @ theta) - y).T @ x[:, i] + lamb / m * theta[i]
    grad[1:theta.shape[0]] = 1 / m * ((sigmoid(x @ theta) - y).T @ x[:, 1:theta.shape[0]]).T + lamb / m * theta[
                                                                                                          1:theta.shape[
                                                                                                              0]];
    return grad.flatten()


def predict_one_vs_all(theta, x):
    """
    :param theta: 权值（优化后的）
    :param x: 测试变量
    :return: 预测值
    """
    m = x.shape[0]
    p = np.zeros((m, 1))
    cal = sigmoid(x @ theta)
    p = np.argmax(cal, axis=1)
    p[p == 0] = 10
    return p


def multi_regression():
    # 导入数据
    print("loading data...")
    data = scio.loadmat("ex3data1.mat")
    x = data['X']
    y = data['y']
    # 训练样本数
    m = x.shape[0]
    # 选取100个样本进行数据展示，洗牌操作
    arr = np.arange(m)
    np.random.shuffle(arr)
    disp = x[arr[0:100], :]
    # 展示数据
    display_data(disp)
    b = np.ones((m, 1), dtype=int)
    x = np.c_[b, x]
    all_theta = np.zeros((x.shape[1], 10))
    # 检验
    print('Testing CostFunction with regularization');

    theta_t = np.array([[-2], [-1], [1], [2]])
    X_t = np.c_[
        np.ones((5, 1)), np.array([[0.1, 0.6, 1.1], [0.2, 0.7, 1.2], [0.3, 0.8, 1.3], [0.4, 0.9, 1.4], [0.5, 1, 1.5]])];
    y_t = (np.array([[1], [0], [1], [0], [1]]) >= 0.5);
    lambda_t = 3;
    J = cost_function(theta_t, X_t, y_t, lambda_t);
    g = grad(theta_t, X_t, y_t, lambda_t)
    print('Cost: %f', J);
    print('Expected cost: 2.534819\n');
    print('Gradients:\n');
    print(' %f \n', g);
    print('Expected gradients:\n');
    print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');
    # 分别训练
    lamb = 0.1
    for i in range(0, 10):
        test = y.copy()
        if i == 0:
            test[test != 10] = 0
            test[test == 10] = 1
        else:
            test[test != i] = 0
            test[test == i] = 1
        initial_theta = np.zeros((x.shape[1], 1))
        result = op.minimize(fun=cost_function, x0=initial_theta.flatten(), args=(x, test, lamb), method='TNC',
                             jac=grad)
        theta = result.x
        all_theta[:, i] = theta
    # 预测
    p = predict_one_vs_all(all_theta, x)
    count = 0
    for i in range(p.shape[0]):
        if p[i] == y[i][0]:
            count += 1
    print('Train Accuracy:', count / m)


multi_regression()
