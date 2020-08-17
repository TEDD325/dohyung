import random
import matplotlib.pyplot as plt
from collections import Counter
import math
import numpy as np


def mean(x):
    return sum(x) / len(x)


def median(x):
    n = len(x)
    sorted_x = sorted(x)
    m = n // 2
    if n % 2 == 1:  # if odd
        return sorted_x[m]  # the middle value
    else:  # if even
        return (sorted_x[m - 1] + sorted_x[m]) / 2  # the average of the middle values


def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]


def variance(x):
    m = mean(x)
    return sum([(x_i - m) ** 2 for x_i in x]) / len(x)


def standard_deviation(x):
    return math.sqrt(variance(x))


if __name__ == "__main__":
    n = 30
    random.seed(0)
    x = [math.floor(10 * (math.log(1 + random.random() * 1023, 2))) / 10  # list of 30 random values
         for __ in range(n)]
    print("values:", x)
    
    print("count:", len(x))
    print("maximum:", max(x))
    print("minimum:", min(x))
    
    x_sorted = sorted(x)
    print("sorted:", x_sorted)
    print("minimum:", x_sorted[0])
    print("maximum:", x_sorted[-1])
    
    print("mean:", mean(x))
    
    print("median:", median(x))
    
    print("mode:", mode(x))
    
    print("variance:", variance(x))
    print("standard_deviation:", standard_deviation(x))
    
    y = [math.floor(10 * (math.log(1 + random.random() * 1023, 2))) / 10  # list of 30 random values
         for __ in range(n)]
    plt.scatter(x, y, color='red', marker='x')  # add a scatter plot
    plt.show()  # show the figure
    
    print("correlation:", np.corrcoef(x, y))
    print("correlation:", np.corrcoef(sorted(x), sorted(y)))