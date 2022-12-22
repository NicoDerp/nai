
from nai.neuralnets import *
from nai.activations import *


class MLP:
    def __init__(self, layers):
        self.net = MLPNeuralNetwork(layers, activation=ReLU)
        self.net.learning_rate = 0.05

    def train(self, dataset, epochs=10, batch_size=50):
        ds = dataset.dimensions[0] * dataset.dimensions[1]
        if ds != self.net.layerSizes[0]:
            raise ValueError(f"Dataset shape {dataset.shape} does not match the neural network's input shape (1, {self.net.layerSizes[0]}).")

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            for batch in range(batch_size):
                self.net.forwardPropagate()

    def _epoch(self, batch_size):
        pass

