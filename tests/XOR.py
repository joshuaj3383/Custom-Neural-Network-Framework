import random
from simple_nn.NeuralNetwork import NeuralNetwork
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction

def create_xor_data(amount):
    '''Randomly generate a set of xor data'''
    xor_data = [
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [[0], [1], [1], [0]]
    ]

    X = []
    Y = []

    for i in range(amount):
        n = random.randint(0,3)
        X.append(xor_data[0][n])
        Y.append(xor_data[1][n])

    return X, Y

def test_nn(nn: NeuralNetwork):
    '''Test the neural network for each possible combination'''
    xor_data = [
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [[0], [1], [1], [0]]
    ]


    for x, y in zip(xor_data[0], xor_data[1]):
        pred = round(nn.predict(x)[0][0],3)

        print(f"Input: {x}, Output: {pred}, Expected: {y[0]}, Passed: {round(pred) == y[0]}")

        #print(f"First Layer activation: {nn.layers[0].forward(x)}")

    #print(nn)

def print_info():
    print("Information==========================")
    print("The XOR problem is a classical problem in machine learning "
          "as it can't be described by a perceptron (single linear neuron)")
    print("XOR takes two truth values [x,y] and returns if only one is true")
    print("Eg: [1,1] (=[true,true]) with return 0 (false) because both are true")
    print("The neural net will return a single value: how likely it is to be true/false")


if __name__ == "__main__":
    print_info()

    #Intrestingly enough, it appears that the sign of the starting biases has
    #a large impact on the result. This applies to the first layer and only the first layer
    #Anything positive will almost always lead to a 4/4 correct while anything <= 0 will often
    #lead to 3/4

    X, Y = create_xor_data(1000)
    layer0 = Layer([2,2],.5,ActivationFunction.SIGMOID)
    layer2 = Layer([2,1],.5,ActivationFunction.SIGMOID)
    layers = [layer0, layer2]

    nn = NeuralNetwork(layers)

    print("\nTraining=================")
    nn.train(X,Y,5,1.5,.9)

    print("\nTesting =================")
    test_nn(nn)







