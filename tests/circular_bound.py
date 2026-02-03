import random
import numpy as np
from simple_nn.NeuralNetwork import NeuralNetwork
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction
import matplotlib.pyplot as plt

def generate_data(amount, radius=4, scale=2):
    """Generates test data X: [x1,x2] Y: [within circular bounds]"""
    X = []
    Y = []

    for i in range(amount):
        x1 = random.uniform(-radius*scale, radius*scale)
        x2 = random.uniform(-radius*scale, radius*scale)
        y = 1 if x1*x1 + x2*x2 < radius*radius else 0

        X.append([x1, x2])
        Y.append([y])

    return np.array(X), np.array(Y)

def test_nn(nn, data_points):
    """ Tests the neural network's ability to predict whether a point is in bounds"""

    X, Y = generate_data(data_points, 4,2)
    Y_pred, _ = nn.test(X, Y)
    Y_pred = [[round(float(y_pred[0][0]))] for y_pred in Y_pred]
    Y_pred = np.array(Y_pred)

    return X, Y, Y_pred


def plot_data(X: np.ndarray, Y_pred: np.ndarray, radius: int):
    """Plots nn's predictions as well as correct bounds"""

    # Sort data into true false
    x_true = X[Y_pred.flatten() == 1]
    x_false = X[Y_pred.flatten() == 0]

    # Plot true and false
    plt.scatter(x_true[:, 0], x_true[:, 1], color='blue', label='True', s=5)
    plt.scatter(x_false[:, 0], x_false[:, 1], color='red', label='False', s=5)

    # Draw reference circle
    theta = np.linspace(0, 2 * np.pi, 300)  # Angle values from 0 to 2π
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)

    plt.plot(x_circle, y_circle, color='black', label='Circle Bounds (r={})'.format(radius))

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('NN Predication bounds vs. Reference Bound')
    plt.legend()
    plt.show()

def count_correct(Y: np.ndarray, Y_pred: np.ndarray) -> int:
    """Counts how many cases the nn got correct"""

    c = 0
    for y1, y2 in zip(Y, Y_pred):
        if y1 == y2:
            c += 1

    return c


if __name__ == '__main__':
    radius = 4
    scale = 2 # I have found that the training scale doesnt matter as long as >1
    train_data_n = 2500
    test_data_n = 7500

    # Create Layers
    layer0 = Layer([2, 25], -1, ActivationFunction.LEAKY_RELU)
    layer1 = Layer([25, 25], 0.5, ActivationFunction.LEAKY_RELU)
    layer2 = Layer([25, 1], 1, ActivationFunction.SIGMOID)
    layers = [layer0, layer1, layer2]

    # Create NN
    nn = NeuralNetwork(layers)

    #Train
    print("Training=================================")
    X, Y = generate_data(train_data_n, radius, scale)
    nn.train(X,Y,10,.1,.95)

    # Test nn
    X, Y, Y_pred = test_nn(nn, test_data_n)

    # Graph
    plot_data(X, Y_pred, radius)

    # Calculate Accuracy
    c = count_correct(Y, Y_pred)
    print(f"\nAccuracy: {c}/{len(Y)} = {round(c/len(Y)*100,3)}%")



