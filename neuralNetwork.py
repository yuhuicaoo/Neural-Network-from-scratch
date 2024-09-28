import numpy as np
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def calc_loss(self, output, target, loss_function=None):
        if loss_function is None:
            raise ValueError("Loss function not provided")
        
        loss = loss_function(output, target)
        batch_loss = np.mean(loss)
        return batch_loss

