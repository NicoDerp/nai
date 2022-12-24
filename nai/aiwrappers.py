
from nai.neuralnets import *
from nai.activations import *

import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layers):
        self.net = MLPNeuralNetwork(layers, activation=Sigmoid)
        self.net.learning_rate = 0.2

    def train(self, dataset, epochs=10, batch_size=32):
        #if dataset.shape != (1, self.net.layerSizes[0]):
        #    raise ValueError(f"Dataset shape {dataset.shape} does not match the neural network's input shape (1, {self.net.layerSizes[0]}).")

        lossArray = []

        #dataset.useSet(SetTypes.Train)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            averageLoss = 0

            dataset.shuffle()

            errorSum = [zero(self.net.layerSizes[i + 1]) for i in range(self.net.nLayers - 1)]

            samples = dataset.retrieveBatch(batch_size)
            print([(sample.data, sample.output) for sample in samples])

            for sample in samples:
                print("\nUsing data:", sample.data)
                self.net.layers[0] = sample.data
                self.net.forwardPropagate()
                print("Got answ:", self.net.layers[-1])

                loss = self.net.calculateLoss()
                averageLoss += loss
                print(f"Loss: {loss:.10f}")

                self.net.expectedOutput = sample.output
                errs = self.net.calcErrors()

                # Add new deltas to sum
                for layer in range(self.net.nLayers - 1):
                    for i in range(len(errs[layer])):
                        errorSum[layer][i] += errs[layer][i]

            for layer in range(self.net.nLayers - 1):
                for i in range(len(errorSum[layer])):
                    errorSum[layer][i] /= batch_size

            self.net.backPropagate(errorSum)

            # Debug
            lossArray.append(averageLoss / batch_size)

        # Debug
        plt.plot(range(epochs), lossArray)
        plt.grid()
        plt.show()

    def test(self, dataset):
        #if dataset.shape != (1, self.net.layerSizes[0]):
        #    raise ValueError(f"Dataset shape {dataset.shape} does not match the neural network's input shape (1, {self.net.layerSizes[0]}).")

        #dataset.useSet(SetTypes.Train)

        dataset.shuffle()

        averageLoss = 0
        nSamples = 500

        for i in range(nSamples):
            sample = dataset.retrieveSample()

            self.net.layers[0] = sample.data
            self.net.forwardPropagate()

            self.net.expectedOutput = sample.output
            averageLoss += self.net.calculateLoss()

        averageLoss /= nSamples
        print(f"Average loss for {nSamples} samples is {averageLoss}")

    def _epoch(self, batch_size):
        pass

