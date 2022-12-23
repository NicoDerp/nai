
from nai.neuralnets import *
from nai.activations import *

import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layers):
        self.net = MLPNeuralNetwork(layers, activation=ReLU)
        self.net.learning_rate = 0.05

    def train(self, dataset, epochs=10, batch_size=32):
        if self.net.shape != (1, self.net.layerSizes[0]):
            raise ValueError(f"Dataset shape {dataset.shape} does not match the neural network's input shape (1, {self.net.layerSizes[0]}).")

        lossArray = []

        dataset.useSet(SetTypes.Train)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            dataset.shuffle()

            dwSum = [zero(self.net.layerSizes[i] * self.net.layerSizes[i + 1]) for i in range(nLayers - 1)]
            dbSum = [zero(self.net.layersSizes[i + 1]) for i in range(self.net.nLayers - 1)]

            for batch in range(batch_size):
                sample = dataset.retrieveSample()

                self.net.layers[0] = sample.data
                self.net.forwardPropagate()

                # Debug
                lossArray.append(self.net.calculateLoss())

                self.net.expectedOutput = sample.output
                dws, dbs = self.net.calcDeltas()

                # Add new deltas to sum
                for layer, (dwL, dbL) in enumerate(zip(dws, dbs)):
                    for i, dw, in enumerate(dwL):
                        dwSum[layer][i] += dw

                    for i, db in enumerate(dbL):
                        dbSum[layer][i] += db

            # Update weights and biases with average delts
            for layer, (dwL, dbL) in enumerate(zip(dws, dbs)):
                for i, dw, in enumerate(dwL):
                    self.net.weights[layer][i] += dwSum[layer][i] / batch_size

                for i, db in enumerate(dbL):
                    self.net.biases[layer][i] += dbSum[layer][i] / batch_size

        # Debug
        plt.plot(range(epochs*batch_size), lossArray)
        plt.grid()
        plt.legend()
        plt.show()

    def _epoch(self, batch_size):
        pass

