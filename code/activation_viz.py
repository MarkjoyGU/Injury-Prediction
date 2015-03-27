import math
import numpy as np
import matplotlib.pyplot as plt

@np.vectorize
def sigmoid(x):
    return 1 / (1 + math.exp(-2 * x))


@np.vectorize
def cosine(x):
    return (math.cos(x) / 2) + 0.5


@np.vectorize
def sine(x):
    return (math.sin(x) / 2) + 0.5


@np.vectorize
def gaussian(x):
    return 1 / (math.exp(x**2))


@np.vectorize
def elliot(x):
    return ((0.5 * x) / (1 + abs(x))) + 0.5


@np.vectorize
def linear(x):
    if x > 1:
        return 1
    return 0


@np.vectorize
def threshold(x):
    if x < 0:
        return 0
    return 1

if __name__ == '__main__':
    x = np.linspace(-4, 4, 100)
    # plt.plot(x, sigmoid(x), label='Sigmoid')
    # plt.plot(x, cosine(x), label='Cosine')
    # plt.plot(x, sine(x), label='Sine')
    plt.plot(x, gaussian(x), label='Gaussian')
    plt.plot(x, elliot(x), label='Elliot')
    plt.plot(x, linear(x), label='Linear')
    plt.plot(x, threshold(x), label='Threshold')

    plt.legend(loc='upper left')
    plt.xlabel('Weighted Sum')
    plt.ylabel('Output to Next Layer')

    plt.show()
