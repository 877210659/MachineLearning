# !/usr/bin/env python 
# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
# 拟合逻辑回归模型，解决分类问题
# 使用正则化，使用梯度下降算法
# 作者：胡森
import numpy as np
import matplotlib.pyplot as plt


# sigmoid函数（logistic函数）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_decision_boundary(theta, x, y):
    """
    :param theta: theta值 3*1 
    :param x: 包含一列1的输入变量矩阵 m*3
    :param y: 输出变量矩阵 m*1
    """
    # 先绘制出散点图
    plot_data(x[:, 1:3], y)
    if x.shape[1] <= 3:
        # 当变量只有两个
        plot_x = x[:, 1]
        plot_y = -1 / theta[2][0] * (theta[0][0] + theta[1][0] * plot_x)
        plt.title("decision boundary plot")
        plt.xlabel("Exam1 score")
        plt.ylabel("Exam2 score")
        plt.plot(plot_x, plot_y, label="boundary")
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        u.shape = u.size, 1
        v.shape = v.size, 1
        z = np.zeros((u.shape[0], v.shape[0]))
        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                z[i][j] = map_feature(u[i][0], v[j][0]) @ theta
        u, v = np.meshgrid(u, v)
        # 此时z需要转置，原理和线性回归时一样
        plt.contour(u, v, z.T, [0], colors='k')


def map_feature(x1, x2):
    """
    :param x1: variable one 
    :param x2: variable two
    :return: 新的特征量，用于构造多项式拟合曲线
    """
    # 最高6阶
    degree = 6
    # 添加一列1
    out = np.ones((x1.size, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            b = pow(x1, i - j) * pow(x2, j)
            out = np.c_[out, b]
    return out


def gradient_reg(theta, x, y, lamb):
    """
    :param theta: theta值 3*1 
    :param x: 包含一列1的输入变量矩阵 m*3
    :param y: 输出变量矩阵 m*1
    :param lamb: 正则化参数
    :return: 梯度算子
    """
    # 样本数
    m, n = x.shape
    theta = theta.reshape((n, 1))
    grad = np.zeros((n, 1))
    grad[0] = 1 / m * (sigmoid(x @ theta) - y).T @ x[:, 0]
    for i in range(1, n):
        grad[i] = 1 / m * (sigmoid(x @ theta) - y).T @ x[:, i] + lamb / m * theta[i]
    return grad


def cost_function_reg(theta, x, y, lamb):
    """
    :param theta: theta值 3*1 
    :param x: 包含一列1的输入变量矩阵 m*3
    :param y: 输出变量矩阵 m*1
    :param lamb: 正则化参数
    :return: 代价函数值
    """
    # 样本数
    m, n = x.shape
    theta = theta.reshape((n, 1))
    cost = -1 / m * np.sum(np.log(sigmoid(x @ theta)) * y + (1 - y) * np.log(1 - sigmoid(x @ theta))) + lamb / (
        2 * m) * (np.sum(theta * theta) - theta[0] * theta[0]);
    return cost


# gradient descent
def gradient_descent(x, y, alpha, theta, num_iterations, lamb):
    """
    :param x: the matrix of independent variables 
    :param y: the vector of sample value
    :param alpha: learning rate 
    :param theta: the vector of cost function's parameters
    :param num_iterations: the num of iterations
    :param lamb: 正则化参数
    :return: theta,J_history
    """
    # Initialize some useful values
    j_history = np.zeros((num_iterations, 1))
    for i in range(num_iterations):
        theta = theta - alpha * gradient_reg(theta, x, y, lamb)
        j_history[i][0] = cost_function_reg(theta, x, y, lamb)
    return theta, j_history


def plot_data(x, y):
    """
    :param x: m*2矩阵（两个变量） 
    :param y: m*1矩阵
    """
    plt.figure()
    # 大小
    sz = 40
    plt.title("data scatter plot")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], sz, marker='+', color='red', label="Admitted")
    plt.scatter(x[np.where(y == 0), 0], x[np.where(y == 0), 1], sz, marker='o', color='blue', label="Not Admitted")


def predict(theta, x):
    k = np.where(x @ theta >= 0)
    p = np.zeros((x.shape[0], 1))
    for i in k[0]:
        p[i][0] = 1
    return p


def logistic_regression3(filename):
    # 引入数据文件
    print("plotting data...\n")
    data = np.loadtxt(filename, delimiter=',')
    # 样本数
    m = data.shape[0]
    # 变量数
    x = data[:, 0:2]
    y = data[:, 2]
    y.shape = m, 1
    # 绘制散点图
    plot_data(x, y)
    plt.show()
    plt.legend(loc="upper right")
    # 拟合多项式
    x = map_feature(x[:, 0], x[:, 1])
    # 测试：计算代价函数和梯度算子
    initial_theta = np.zeros((x.shape[1], 1))
    # 设置正则化参数
    lamb = 1
    cost = cost_function_reg(initial_theta, x, y, lamb)
    grad = gradient_reg(initial_theta, x, y, lamb)
    print('Cost at initial theta (zeros): ', cost)
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros): ')
    print(grad)
    print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')
    # 使用梯度下降算法
    alpha = 0.01
    iterations = 100000
    theta, j_history = gradient_descent(x, y, alpha, initial_theta, iterations, lamb)
    print(theta)
    # 绘制梯度图
    plt.plot(range(iterations), j_history)
    plt.show()
    # 绘制决策边界
    plot_decision_boundary(theta, x, y)
    plt.legend(loc="upper right")
    plt.show()
    # 检验样本的准确度
    p = predict(theta, x)
    count = 0
    for i in range(p.shape[0]):
        if p[i][0] == y[i][0]:
            count += 1
    print('Train Accuracy:', count / m)


logistic_regression3("ex2data2.txt")