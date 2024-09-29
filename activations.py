import numpy as np


def sigmoid(x, deriv=False):
    sig = 1 / (1 + np.exp(-x))
    return (sig * (1 - sig)) if deriv else sig

def tanh(x, deriv=False):
    return (1 - np.tanh(x)**2) if deriv else np.tanh(x)


def ReLU(x, deriv=False):
    return np.where(x > 0, 1, 0) if deriv else np.maximum(x, 0)

def leaky_ReLU(x, alpha=0.01, deriv=False):
    return np.where(x > 0, 1, alpha) if deriv else np.where(x > 0, 1, alpha * x)


def soft_max(x):
    # subtract max value from the vector to prevent overflow 
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    # normalise to return proababilites
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
