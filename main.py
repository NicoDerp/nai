#!/usr/bin/python3

from nai import *
import numpy as np

dataset = datasets.MNIST("datasets", download=True)
#dataset = datasets.XOR()

model = MLPNeuralNetwork([784, 32, 10], CrossEntropy, ReLU, 0.01)
#model = aiwrappers.MLP([2, 3, 1], MSE, ReLU)

model.train(dataset, epochs=4, batch_size=32)
#model.train(dataset, epochs=4000, batch_size=1)

#model.test(dataset)

print("\n\nTest:")

dataset.shuffle()

np.set_printoptions(suppress=True)

for i in range(10):

    sample = dataset.retrieveSample()
    print("Expected", sample.output)

    model.layers[0] = sample.data

    model.forwardPropagate()
    print(f"Got {model.net.layers[-1]}")

    #continue

    pred = np.argmax(model.layers[-1])
    prob = model.layers[pred]

    print(f"Predicted {pred} with probability of {prob}\n")

