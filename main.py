import numpy as np
from layer import Layer_Dense
from neuralNetwork import NeuralNetwork
from activations import *
from losses import regressionMSE, categoricalCrossEntropy

def main():
    inputs = np.random.randn(1,2)
    nn = NeuralNetwork()
    nn.add_layer(Layer_Dense(2, 4, activation=ReLU))
    nn.add_layer(Layer_Dense(4, 3, activation=soft_max))
    outputs = nn.forward(inputs)
    print(outputs, np.sum(outputs))



if __name__ == "__main__":
    # main()

    X = np.array([[1, 2, 3]])
    y_true = np.array([1,1])  # True class indices

    nn = NeuralNetwork()
    nn.add_layer(Layer_Dense(3,5))
    nn.add_layer(Layer_Dense(5,2))
    output = nn.forward(X)
    print(f"Ouput: \n {output}")
    loss = nn.calc_loss(output, y_true, loss_function=regressionMSE)
    output_grad =(2 * (output - y_true)) / (output.shape[0])
    print(output_grad)
    print(f"Loss: {loss}")

