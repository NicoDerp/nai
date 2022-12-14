#!/usr/bin/python3

from nai import *
import numpy as np

dataset = datasets.MNIST("datasets", download=True)
#dataset = datasets.XOR()

model = MLPNeuralNetwork([784, 32, 10], CrossEntropy, ReLU, 0.01)
#model = aiwrappers.MLP([2, 3, 1], MSE, ReLU)

model.train(dataset, epochs=2, batch_size=32)
#model.train(dataset, epochs=4000, batch_size=1)

#model.test(dataset)

print("\n\nTest:")

dataset.shuffle()

np.set_printoptions(suppress=True)

for i in range(10):

    sample = dataset.retrieveSample()
    print("Expected", sample.output)

    model.net.layers[0] = sample.data

    model.net.forwardPropagate()
    print(f"Got {model.net.layers[-1]}")

    #continue

    pred = np.argmax(model.net.layers[-1])
    prob = model.net.layers[pred]

    print(f"Predicted {pred} with probability of {prob}\n")


exit()

# o - o
#   x
# o - o

net = MLPNeuralNetwork([2, 2, 2], learning_rate=0.1, activation=activations.TanH)
net.layers[0] = np.array([1.0, 4.0])
#net.weights = [np.array([[0.5, 1.0], [2.0, 3.0]]), np.array([[3.0, 2.0], [1.0, 0.5]])]
#net.biases = np.array([[0.1, 0.2], [0.3, 0.4]])

net.expectedOutput = np.array([0.8, 0.5])

print("Trying to get:", net.expectedOutput)

max_epochs = 1000
epoch = 0

#for epoch in range(100):
while epoch < max_epochs:
    loss = net.calculateLoss()
    if loss < 0.0000001:
        break

    print(f"Epoch {epoch+1}/{max_epochs}")

    net.forwardPropagate()
    net.backPropagateError()
    net.gradientDescent()

    print(f"\n{net}\n")
    print(f"Loss {loss:.20f}")

    epoch += 1


