import numpy as np
from layer import Layer_Dense
from neuralNetwork import NeuralNetwork
from activations import *
from losses import regressionMSE, categoricalCrossEntropy

def main():
    pass

if __name__ == "__main__":
    # main()

    X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2))
    y = np.reshape([[0],[1],[1],[0]], (4,1))

    nn = NeuralNetwork()
    nn.add_layer(Layer_Dense(2,4))
    nn.add_layer(Layer_Dense(4,1))
    output = nn.forward(X)

