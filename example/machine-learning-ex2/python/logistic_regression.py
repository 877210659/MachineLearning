# -*- coding:utf-8 -*-
# 拟合逻辑回归模型，解决分类问题
# 不使用正则化，使用高级优化算法
# 作者：胡森
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


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
        z = np.zeros((u.shape[0], v.shape[0]))
        for i in range(u.shape[0]):
            for j in range(v.shape[0]):
                z[i][j] = map_feature(u[i], v[j]) * theta
        z = z.T
        plt.contour(u, v, z, [0, 0], cmap='rainbow')


def map_feature(x1, x2):
    """
    :param x1: variable one 
    :param x2: variable two
    :return: 新的特征量，用于构造多项式拟合曲线
    """
    # 最高6阶
    degree = 6
    # 添加一列1
    out = np.ones((x1.shape[0], 1))
    for i in range(1, degree + 1):
        for j in range(i):
            b = pow(x1, i - j) * pow(x2, j)
            out = np.c_[out, b]
    return out


def gradient(theta, x, y):
    """
    :param theta: theta值 3*1 
    :param x: 包含一列1的输入变量矩阵 m*3
    :param y: 输出变量矩阵 m*1
    :return: 梯度算子
    """
    # 样本数
    m, n = x.shape
    theta = theta.reshape((n, 1))
    grad = 1 / m * x.T @ (sigmoid(x @ theta) - y)
    return grad.flatten()


def cost_function(theta, x, y):
    """
    :param theta: theta值 3*1 
    :param x: 包含一列1的输入变量矩阵 m*3
    :param y: 输出变量矩阵 m*1
    :return: 代价函数值
    """
    # 样本数
    m, n = x.shape
    theta = theta.reshape((n, 1))
    cost = -1 / m * np.sum(np.log(sigmoid(x @ theta)) * y + (1 - y) * np.log(1 - sigmoid(x @ theta)))
    return cost


def plot_data(x, y):
    """
    :param x: m*2矩阵（两个变量） 
    :param y: m*1矩阵
    """
    plt.figure()
    # 大小
    sz = 40
    plt.title("data scatter plot")
    plt.xlabel("Exam1 score")
    plt.ylabel("Exam2 score")
    plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], sz, marker='+', color='red', label="Admitted")
    plt.scatter(x[np.where(y == 0), 0], x[np.where(y == 0), 1], sz, marker='o', color='blue', label="Not Admitted")


def predict(theta, x):
    k = np.where(x @ theta >= 0)
    p = np.zeros((x.shape[0], 1))
    for i in k[0]:
        p[i][0] = 1
    return p


def logistic_regression(filename):
    # 引入数据文件
    print("plotting data...\n")
    data = np.loadtxt(filename, delimiter=',')
    # 样本数
    m = data.shape[0]
    # 变量数
    n = data.shape[1] - 1
    x = data[:, 0:2]
    y = data[:, 2]
    y.shape = m, 1
    # 绘制散点图
    plot_data(x, y)
    plt.show()
    plt.legend(loc="upper right")
    b = np.ones((m, 1), dtype=int)
    x = np.c_[b, x]
    # 测试：计算代价函数和梯度算子
    initial_theta = np.zeros((n + 1, 1))
    cost = cost_function(initial_theta, x, y)
    grad = gradient(initial_theta, x, y)
    print('Cost at initial theta (zeros): ', cost)
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros): ')
    print(grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
    # 使用高级优化算法
    result = op.minimize(fun=cost_function, x0=initial_theta.flatten(), args=(x, y), method='TNC', jac=gradient)
    print(result)
    theta = result.x
    theta.shape = theta.shape[0], 1
    # 绘制决策边界
    plot_decision_boundary(theta, x, y)
    plt.legend(loc="upper right")
    plt.show()
    # 预测
    prob = sigmoid(np.array([[1, 45, 85]]) @ theta)
    print('For a student with scores 45 and 85, we predict an admission probability of :', prob)
    # 检验样本的准确度
    p = predict(theta, x)
    count = 0
    for i in range(p.shape[0]):
        if p[i][0] == y[i][0]:
            count += 1
    print('Train Accuracy:', count / m)


logistic_regression("ex2data1.txt")
