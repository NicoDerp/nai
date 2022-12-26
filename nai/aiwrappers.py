
from nai.neuralnets import *
from nai.activations import *

import matplotlib.pyplot as plt

import numpy as np

from numba import njit

#@njit(fastmath=True)
def _doBatch(net, dataset, batch_size):
    errorSum = np.array([np.zeros(net.layerSizes[i + 1]) for i in range(net.nLayers - 1)])

    averageLoss = 0

    samples = dataset.retrieveBatch(batch_size)
    #print([(sample.data, sample.output) for sample in samples])

    for sample in samples:
        #print("\nUsing data:", sample.data)
        net.layers[0] = sample.data
        net.forwardPropagate()
        #print("Got answ:", self.net.layers[-1])

        net.expectedOutput = sample.output
        errs = net.calcErrors()

        loss = net.calculateLoss()
        averageLoss += loss
        #print(f"Loss: {loss:.10f}")

        # Add new deltas to sum
        #for layer in range(net.nLayers - 1):
        #    for i in range(len(errs[layer])):
        #        errorSum[layer][i] += errs[layer][i]
        #print("e", errorSum)
        #print(errs)
        errorSum = np.add(errorSum, errs)

    #for layer in range(net.nLayers - 1):
    #    for i in range(len(errorSum[layer])):
    #        errorSum[layer][i] /= batch_size

    errorSum /= len(samples)

    net.backPropagate(errorSum)

    return averageLoss / len(samples)


class MLP:
    def __init__(self, layers, adam=False):

        self.net = MLPNeuralNetwork(layers, 0.2, activation=Sigmoid, adam=adam)

    def train(self, dataset, epochs=10, batch_size=32):
        #if dataset.shape != (1, self.net.layerSizes[0]):
        #    raise ValueError(f"Dataset shape {dataset.shape} does not match the neural network's input shape (1, {self.net.layerSizes[0]}).")

        #dataset.useSet(SetTypes.Train)

        nBatches = math.ceil(dataset.size / batch_size)
        print(f"Doing {nBatches} batches per epoch")

        lossArray: List[float] = np.empty(epochs*nBatches)

        batch = 0
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            dataset.shuffle()

            for batch in range(nBatches):
                average_loss = _doBatch(self.net, dataset, batch_size)
                lossArray[batch] = average_loss
                batch += 1

            # Debug
            #lossArray.append(averageLoss / batch_size)

            # Optimizations
            #if self.adam:
            #    Mht = Mt / (1 / self.bias1 ** epoch)
            #    self.net.learning_rate /= math.sqrt(epoch)

        #print(lossArray)

        # Debug
        plt.plot(range(epochs * nBatches), lossArray)
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

