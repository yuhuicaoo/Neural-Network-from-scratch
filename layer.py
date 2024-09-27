import numpy as np
from  utils import *

np.random.seed(42)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, activation=None):
        self.weights = xavier_uniform((n_inputs,n_neurons))
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward(self, inputs):
        # apply linear transformation
        z = np.dot(inputs, self.weights) + self.biases
        # apply activation function if exist else return raw output
        return self.activation(z) if self.activation else z
    