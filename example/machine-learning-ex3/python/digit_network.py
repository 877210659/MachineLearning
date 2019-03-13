# !/usr/bin/env python 
# -*- coding:utf-8 -*-
# 神经网络拟合回归模型，解决多分类问题
# 使用正则化，使用高级优化算法
# 作者：胡森
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import math


# sigmoid函数（logistic函数）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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


def predict(theta1, theta2, x):
    """
    :param theta1:  权重一
    :param theta2: 权重二
    :param x: 输入变量
    :return: 预测数据
    """
    m = x.shape[0]
    b = np.ones((m, 1), dtype=int)
    x = np.c_[b, x]
    A = sigmoid(theta1 @ (x.T))
    m = A.shape[1]
    b = np.ones((1, m), dtype=int)
    A = np.r_[b, A]
    p = sigmoid(theta2 @ A)
    p = p.T
    p = np.argmax(p, axis=1)+1
    return p


def digit_network():
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

    # 导入权重数据
    weights = scio.loadmat("ex3weights.mat")
    # 预测
    p = predict(weights['Theta1'], weights['Theta2'], x)
    count = 0
    for i in range(p.shape[0]):
        if p[i] == y[i][0]:
            count += 1
    print('Train Accuracy:', count / m)
    for i in arr:
        print('Displaying Example Image\n');
        test = x[arr[i], :]
        test.shape = 1,test.size
        display_data(test);
        pred = predict(weights['Theta1'], weights['Theta2'],test);
        print('Neural Network Prediction:', pred);
        s = input('Paused - press enter to continue, q to exit:');
        if s == 'q':
            break


digit_network()
