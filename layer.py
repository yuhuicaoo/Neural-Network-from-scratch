import numpy as np
from utils import *

np.random.seed(42)


class Layer_Dense:
    def __init__(self, n_inputs, n_outputs):
        # self.weights = xavier_uniform((n_outputs, n_inputs))
        # self.biases = np.zeros((n_outputs,1))
        self.weights = np.random.randn(n_outputs, n_inputs)
        self.biases = np.random.rand(n_outputs,1)


    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.biases

    def backward(self, output_grads):
        pass