
from nai.neuralnets import *
from nai.activations import *

import matplotlib.pyplot as plt

import numpy as np

from numba import njit

#@njit(fastmath=True)
def _doBatch(net, dataset, batch_size):
    averageError = np.array([np.zeros(net.layerSizes[i + 1]) for i in range(net.nLayers - 1)], dtype=object)

    averageLoss = 0
    averageAcc = 0

    samples = dataset.retrieveBatch(batch_size)
    #print([(sample.data, sample.output) for sample in samples])

    for sample in samples:
        #print("Sample")
        #print("\nUsing data:", sample.data)
        net.layers[0] = sample.data
        net.forwardPropagate()
        #print("Got answ:", self.net.layers[-1])

        net.expectedOutput = sample.output
        net.backPropagateError()
        averageError = np.add(averageError, net.errors)

        loss = net.calculateLoss()
        averageLoss += loss

        biggest = 0
        biggest_i = 0
        for i, n in enumerate(net.layers[-1]):
            if n > biggest:
                biggest = n
                biggest_i = i

        #averageAcc += 1 if biggest_i == list(sample.output).index(1) else 0

        #print(f"Loss: {loss:.10f}")

        # Add new deltas to sum
        #for layer in range(net.nLayers - 1):
        #    for i in range(len(errs[layer])):
        #        errorSum[layer][i] += errs[layer][i]
        #print("e", errorSum)
        #print(errs)

    #for layer in range(net.nLayers - 1):
    #    for i in range(len(errorSum[layer])):
    #        errorSum[layer][i] /= batch_size

    # Calculate average
    averageError /= len(samples)

    net.errors = averageError
    net.gradientDescent()

    return averageLoss/len(samples), averageAcc/len(samples)


class MLP:
    def __init__(self, layers, adam=False):

        self.net = MLPNeuralNetwork(layers, 0.001, activation=ReLU, adam=adam)

    def train(self, dataset, epochs=10, batch_size=32):
        if batch_size > dataset.size:
            raise ValueError(f"Batch size is greater than dataset size")

        #if dataset.shape != (1, self.net.layerSizes[0]):
        #    raise ValueError(f"Dataset shape {dataset.shape} does not match the neural network's input shape (1, {self.net.layerSizes[0]}).")

        #dataset.useSet(SetTypes.Train)

        nBatches = math.ceil(dataset.size / batch_size)
        print(f"Doing {nBatches} batches per epoch")

        #lossArray = np.empty(epochs*nBatches)
        lossArray = np.empty(epochs)
        #accuracyArray = np.empty(epochs*nBatches)

        #batchCount = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            dataset.shuffle()

            averageLoss = 0

            for batch in range(nBatches):
                loss, acc = _doBatch(self.net, dataset, batch_size)

                averageLoss += loss

                #accuracyArray[batchCount] = acc
                #batchCount += 1

            lossArray[epoch] = averageLoss

            # Debug
            #lossArray.append(averageLoss / batch_size)

            # Optimizations
            #if self.adam:
            #    Mht = Mt / (1 / self.bias1 ** epoch)
            #    self.net.learning_rate /= math.sqrt(epoch)

        print(lossArray)

        # Debug
        #plt.plot(range(epochs * nBatches), lossArray, label="Loss")
        plt.plot(range(epochs), lossArray, label="Loss")
        #plt.plot(range(epochs * nBatches), accuracyArray, label="Accuracy")
        plt.grid()
        plt.legend()
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

