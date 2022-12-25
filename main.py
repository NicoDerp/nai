#!/usr/bin/python3

from nai import *

dataset = datasets.MNIST("datasets", download=True)
#dataset = datasets.XOR()

model = aiwrappers.MLP([784, 32, 10])
#model = aiwrappers.MLP([2, 2, 1])

model.train(dataset, epochs=50, batch_size=32)

#model.test(dataset)

print("\n\nTest:")

dataset.shuffle()

for i in range(5):

    sample = dataset.retrieveSample()
    print(sample.output)

    model.net.layers[0] = sample.data

    #model.net.layers[0] = [1, 0]
    model.net.forwardPropagate()
    #print(model.net.layers[-1])

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


