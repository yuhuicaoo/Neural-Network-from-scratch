import numpy as np
from utils import *

np.random.seed(42)


class Layer_Dense:
    def __init__(self, n_inputs, n_outputs):
        self.weights = xavier_uniform((n_inputs, n_outputs))
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        # keep inputs for backward propagation.
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.biases

    def backward(self, output_grads):
        self.w_grads = np.dot(output_grads, self.inputs)
        self.b_grads = output_grads
        self.x_grads = np.dot(self.weights, output_grads)
        return self.x_grads
