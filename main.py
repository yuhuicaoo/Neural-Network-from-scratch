from layer import Layer_Dense
from neuralNetwork import NeuralNetwork
from activations import *

def main():
    inputs = np.random.randn(1,2)
    print(inputs.T)
    nn = NeuralNetwork()
    nn.add_layer(Layer_Dense(2, 4, activation=ReLU))
    print(nn.layers[0].weights)
    nn.add_layer(Layer_Dense(4, 1, activation=None))
    print(nn.layers[1].weights)
    output = nn.forward(inputs)
    print(output)

if __name__ == "__main__":
    main()
