# -*- coding:utf-8 -*-
# Use the gradient descent to fit the linear regression model, including one or multi variable.
# author:hu sen
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# cost function
def compute_cost(x, y, theta):
    """
    :param x: the matrix of independent variables 
    :param y: the vector of sample value
    :param theta: the vector of cost function's parameters 
    :return: the value of cost function
    """
    # Initialize some useful values
    # 样本数
    m = y.shape[0]
    j = np.sum((x.dot(theta) - y) * (x.dot(theta) - y)) / (2 * m)
    return j


# gradient descent
def gradient_descent(x, y, alpha, theta, num_iterations):
    """
    :param x: the matrix of independent variables 
    :param y: the vector of sample value
    :param alpha: learning rate 
    :param theta: the vector of cost function's parameters
    :param num_iterations: the num of iterations
    :return: theta,J_history
    """
    # Initialize some useful values
    m = y.shape[0]
    j_history = np.zeros((num_iterations, 1))
    temp_list = np.zeros((theta.shape[0], 1))
    for i in range(num_iterations):
        for j in range(theta.shape[0]):
            test = x[:, j]
            test.shape = m, 1
            temp = theta[j][0] - alpha * (test.T.dot((x.dot(theta) - y))) / m
            temp_list[j][0] = temp[0][0]
        theta = temp_list
        j_history[i][0] = compute_cost(x, y, theta)
    return theta, j_history


# feature scaling and mean normalization
def feature_normalize(x):
    """
    :param x: the matrix of independent variables,and I will do feature scaling and mean normalization for it 
    :return: the matrix that have been operated, the mean, the standard deviation
    """
    # Initialize some useful values
    m = x.shape[0]
    mu = np.zeros((1, x.shape[1]))
    sigma = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        mean = np.sum(x[:, i]) / m
        stand_deviation = np.std(x[:, i])
        mu[0][i] = mean
        sigma[0][i] = stand_deviation
        x[:, i] = (x[:, i] - mean) / stand_deviation
    return x, mu, sigma


# linear regression model
def one_variable_model(filename):
    # part1:plot data
    print("plotting data...\n")
    data = np.loadtxt(filename, delimiter=',')  # import data
    m = data.shape[0]
    x = data[:, 0]
    variable_num = data.shape[1] - 1
    x.shape = m, variable_num
    y = data[:, 1]
    y.shape = m, 1
    plt.figure()
    plt.title("data scatter diagram")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y, marker='x', color='blue')
    # part2:Cost and gradient descent
    # no need to operate feature_normalize for single variable linear regression model
    b = np.ones((m, 1), dtype=int)
    x = np.c_[b, x]
    # the settings of gradient descent
    alpha = 0.01
    iterations = 1500
    # initialize
    theta = np.zeros((variable_num + 1, 1))
    # run gradient descent
    [theta, j_history] = gradient_descent(x, y, alpha, theta, iterations)
    print(theta)
    # plot the linear fit
    plt.plot(x[:, 1], x.dot(theta))
    plt.show()
    plt.title("the value of cost function")
    plt.xlabel("num of iterations")
    plt.ylabel("value")
    plt.plot(range(iterations), j_history)
    plt.show()
    # Predict values
    # for population sizes of 35, 000 and 70, 000
    predict1 = [1, 3.5] @ theta
    print('For population = 35,000, we predict a profit of: ', predict1 * 10000)
    predict2 = [1, 7] @ theta
    print('For population = 70,000, we predict a profit of: ', predict2 * 10000)
    # part3 visualizing cost function(one variable)
    print('Visualizing J(theta_0, theta_1) ...\n')
    theta0_values = np.linspace(-10, 10, 100)
    theta1_values = np.linspace(-1, 4, 100)
    # init
    j_values = np.zeros((theta0_values.shape[0], theta1_values.shape[0]))
    for i in range(theta0_values.shape[0]):
        for j in range(theta1_values.shape[0]):
            theta2 = np.array([[theta0_values[i]], [theta1_values[j]]])
            # need transpose
            j_values[j][i] = compute_cost(x, y, theta2)
    fig = plt.figure()
    ax = Axes3D(fig)
    theta0, theta1 = np.meshgrid(theta0_values, theta1_values)
    plt.title("visualizing cost function")
    plt.xlabel("theta0")
    plt.ylabel("theta1")
    ax.plot_surface(theta0, theta1, j_values, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
    plt.title("visualizing cost function")
    plt.xlabel("theta0")
    plt.ylabel("theta1")
    plt.contour(theta0, theta1, j_values, np.logspace(-2, 3, 20), cmap='rainbow')
    plt.scatter(theta[0][0], theta[1][0], marker='x', color='blue')
    plt.show()


def multi_variable_model(filename):
    data = np.loadtxt(filename, delimiter=',')  # import data
    m = data.shape[0]
    variable_num = data.shape[1] - 1
    x = data[:, 0:variable_num]
    y = data[:, variable_num]
    y.shape = m, 1
    # Scale features and set them to zero mean
    print("feature scaling and mean normalization...")
    [x, mu, sigma] = feature_normalize(x)
    b = np.ones((m, 1), dtype=int)
    x = np.c_[b, x]
    # the settings of gradient descent
    alpha = 0.01;
    iterations = 1000;
    # initialize
    theta = np.zeros((variable_num + 1, 1))
    # run gradient descent
    [theta, j_history] = gradient_descent(x, y, alpha, theta, iterations)
    print(theta)
    # plot the linear fit
    plt.title("the value of cost function")
    plt.xlabel("num of iterations")
    plt.ylabel("value")
    plt.plot(range(iterations), j_history)
    plt.show()
    # Predict values
    # two variables
    x1 = (1650 - mu[0][0]) / sigma[0][0]
    x2 = (3 - mu[0][1]) / sigma[0][1]
    test = np.array([[1,x1 ,x2 ]])
    price = test @ theta;

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

    print(['Predicted price of a 1650 sq-ft, 3 br house '
           '(using gradient descent): '], price);



one_variable_model("ex1data1.txt")