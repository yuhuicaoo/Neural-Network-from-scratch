import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ReLU(x):
    return np.maximum(x,0)

def leaky_ReLU(x):
    return np.maximum(0.1*x,x)


