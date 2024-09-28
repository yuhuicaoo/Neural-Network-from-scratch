import numpy as np
from layer import Layer_Dense
from neuralNetwork import NeuralNetwork
from activations import *
from losses import *

def main():
    inputs = np.random.randn(1,2)
    nn = NeuralNetwork()
    nn.add_layer(Layer_Dense(2, 4, activation=ReLU))
    nn.add_layer(Layer_Dense(4, 3, activation=soft_max))
    outputs = nn.forward(inputs)
    print(outputs, np.sum(outputs))



if __name__ == "__main__":
    # main()

    X = np.array([[1, 2, 3], [4, 5, 6]])
    y_true = np.array([1,1])  # True class indices

    nn = NeuralNetwork()
    nn.add_layer(Layer_Dense(3,5, activation=ReLU))
    nn.add_layer(Layer_Dense(5,2, activation=soft_max))
    output = nn.forward(X)
    print(f"Ouput: {output}")
    loss = nn.calc_loss(output, y_true, loss_function=categoricalCrossEntropy)
    print(f"Loss: {loss}")

