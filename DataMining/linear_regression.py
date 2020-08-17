import random
import matplotlib.pyplot as plt
from least_squares_fit import error, linear_model, sum_squared_errors


def gradient_descent(objective_ftn, gradient_ftn, xy, parameters_0, step_size_0=0.01):
    parameters = parameters_0  # initial parameters
    step_size = step_size_0  # initial step size
    min_parameters, min_value = None, float("inf")  # the minimum so far
    iterations_without_improvement = 0
    while iterations_without_improvement < 100:  # if no improvement for 100 iterations, then stop
        value = sum(objective_ftn(xy_i, parameters) for xy_i in xy)
        if value < min_value:  # if new minimum
            min_parameters, min_value = parameters, value
            iterations_without_improvement = 0  # reset
            step_size = step_size_0
        else:  # if no improvement
            iterations_without_improvement += 1
            step_size *= 0.9  # decrease the step size
        for xy_i in shuffle(xy):  # for each data point
            gradient_i = gradient_ftn(xy_i, parameters)  # find new parameters
            parameters = [c - step_size * g  # params = params - step_size * gradient
                            for c, g in zip(parameters, gradient_i)]
    return min_parameters


def shuffle(xy):
    indexes = [i for i, _ in enumerate(xy)]  # create a list of indexes
    random.shuffle(indexes)  # shuffle them
    for i in indexes:  # return the data in that order
        yield xy[i]


if __name__ == "__main__":
    random.seed(0)
    x = range(100);
    y = [linear_model(1, 1, x_i) + random.random() * 5 for x_i in x]
    plt.scatter(x, y)
    plt.show()

    parameters = [random.random(), random.random()]
    
    def squared_error(xy_i, parameters):
        x_i = xy_i[0]
        y_i = xy_i[1]
        alpha, beta = parameters
        return error(alpha, beta, x_i, y_i) ** 2

    def squared_error_gradient(xy_i, parameters):
        x_i = xy_i[0]
        y_i = xy_i[1]
        alpha, beta = parameters
        return [-2 * error(alpha, beta, x_i, y_i),  # partial derivatives
                -2 * error(alpha, beta, x_i, y_i) * x_i]

    alpha, beta = gradient_descent(squared_error, squared_error_gradient, list(zip(x, y)), parameters, 0.0001)
    print("parameters:", alpha, beta)
    print("sum of squared errors:", sum_squared_errors(alpha, beta, x, y))
    
    plt.scatter(x, y)
    plt.plot(x, [linear_model(alpha, beta, x_i) for x_i in x], color='red')
    plt.show()