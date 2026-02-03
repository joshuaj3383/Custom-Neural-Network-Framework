import random
import matplotlib.pyplot as plt
from simple_nn.NeuralNetwork import NeuralNetwork
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction

def create_xor_data():
    xor_data = [
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [[0], [1], [1], [0]]
    ]

    X = []
    Y = []

    for i in range(500):
        n = random.randint(0,3)
        X.append(xor_data[0][n])
        Y.append(xor_data[1][n])

    return X, Y

def test_nn(nn: NeuralNetwork):
    xor_data = [
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [[0], [1], [1], [0]]
    ]

    for x, y in zip(xor_data[0], xor_data[1]):
        pred = nn.predict(x)[0][0]

        print(f"Input: {x}, Output: {pred}, Expected: {y[0]}, Passed: {round(pred) == y[0]}")


layer0 = Layer([2,2],.5,ActivationFunction.RELU)
layer2 = Layer([2,1],1.5,ActivationFunction.NONE)


layers = [layer0, layer2]

nn = NeuralNetwork(layers)

X, Y = create_xor_data()

nn.train(X,Y,5,0.1)

test_nn(nn)

print(nn)


print("repr---------------------------------")
print(nn.__repr__())




