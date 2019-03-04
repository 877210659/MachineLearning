# !/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np


def normal_eqn(x, y):
    theta = np.linalg.inv((x.T @ x)) @ x.T @ y;
    return theta

def model(filename):
    data = np.loadtxt(filename, delimiter=',')  # import data
    m = data.shape[0]
    variable_num = data.shape[1] - 1
    x = data[:, 0:variable_num]
    y = data[:, variable_num]
    y.shape = m, 1
    b = np.ones((m, 1), dtype=int)
    x = np.c_[b, x]
    theta = normal_eqn(x, y)
    print(theta)
    price =  [1,1650,3] @ theta;

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

    print(['Predicted price of a 1650 sq-ft, 3 br house '
           '(using gradient descent): '], price);


model("ex1data2.txt")