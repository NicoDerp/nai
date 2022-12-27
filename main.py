#!/usr/bin/python3

from nai import *
import numpy as np

dataset = datasets.MNIST("datasets", download=True)
#dataset = datasets.XOR()

model = aiwrappers.MLP([784, 32, 10])
#model = aiwrappers.MLP([2, 3, 1])

model.train(dataset, epochs=10, batch_size=10)

#model.test(dataset)

print("\n\nTest:")

dataset.shuffle()

for i in range(5):

    sample = dataset.retrieveSample()
    print(sample.output)

    model.net.layers[0] = sample.data

    model.net.layers[0] = [1, 0]
    model.net.forwardPropagate()
    print(model.net.layers[-1])

    biggest = 0
    biggest_i = 0
    for i, n in enumerate(model.net.layers[-1]):
        if n > biggest:
            biggest = n
            biggest_i = i

    print(f"Predicted {biggest_i} with probability of {biggest}\n")

#model.net.expectedOutput = [1]
#print(model.net.calculateLoss())

exit()

# o - o
#   x
# o - o

net = MLPNeuralNetwork([2, 2, 2], learning_rate=0.1, activation=activations.ActivationFunction)
net.layers[0] = np.array([1, 4])
net.weights = np.array([[0.5, 1, 2, 3], [3, 2, 1, 0.5]])
net.biases = np.array([[0.1, 0.2], [0.3, 0.4]])

net.expectedOutput = [0.8, 0.5]

print("Trying to get:", net.expectedOutput)

max_epochs = 10000
epoch = 0

#for epoch in range(100):
while epoch < max_epochs:
    loss = net.calculateLoss()
    if loss < 0.0000001:
        break

    print(f"Epoch {epoch+1}/{max_epochs}")

    net.forwardPropagate()
    e = net.backPropagateError()
    net.gradientDescent(e)

    print(f"\n{net}\n")
    print(f"Loss {loss:.20f}")

    epoch += 1


