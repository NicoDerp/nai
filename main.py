#!/usr/bin/python3

from nai import *

dataset = datasets.MNIST("datasets", download=True)

model = aiwrappers.MLP([784, 5, 3, 10])

model.train(dataset, epochs=50, batch_size=10)

#model.test(dataset)

exit()

net = neuralnet.MLPNeuralNetwork([2, 3, 2], activation=activations.ReLU)
net.layers[0] = [0.1, 0.3]
net.expectedOutput = [0.8, 0.5]
net.learning_rate = 0.05

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
    net.backPropagate()
    print(f"\n{net}\n")
    print(f"Loss {loss:.20f}")

    epoch += 1


