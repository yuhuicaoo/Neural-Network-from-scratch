import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def ReLU(x):
    return np.maximum(x, 0)


def leaky_ReLU(x):
    return np.maximum(0.1 * x, x)


def soft_max(x):
    # subtract max value from the vector to prevent overflow 
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    # normalise to return proababilites
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
