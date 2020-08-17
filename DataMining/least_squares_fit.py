from analysis import mean, standard_deviation
import matplotlib.pyplot as plt
import random
import numpy as np


def linear_model(alpha, beta, x_i):
    return beta * x_i + alpha


def least_squares_fit(x, y):
    beta = np.corrcoef(x, y)[0, 1] * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def error(alpha, beta, x_i, y_i):
    return y_i - linear_model(alpha, beta, x_i)


def sum_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))


if __name__ == "__main__":
    random.seed(0)
    x = range(100);
    y = [linear_model(1, 1, x_i) + random.random() * 5 for x_i in x]
    plt.scatter(x, y)
    plt.show()
    
    alpha, beta = least_squares_fit(x, y)
    print("parameters:", alpha, beta)
    print("sum of squared errors:", sum_squared_errors(alpha, beta, x, y))
    plt.scatter(x, y)
    y_predicted = [linear_model(alpha, beta, x_i) for x_i in x]
    plt.plot(x, y_predicted, color='red')
    plt.show()