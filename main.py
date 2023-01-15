#!/usr/bin/python3

from nai import *
import numpy as np

dataset = datasets.MNIST("datasets", download=True)
#dataset = datasets.XOR()

model = MLPNeuralNetwork([784, 128, 10],
                         lossfunction=CrossEntropy,
                         activations=[ReLU, Softmax],
                         learning_rate=0.01,
                         dropout=0.2)

#model = aiwrappers.MLP([2, 3, 1], MSE, ReLU)

model.train(dataset, epochs=1, batch_size=64)

#model.train(dataset, epochs=4000, batch_size=1)

model.save("myModel.model")

model.test(dataset)

exit()



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

