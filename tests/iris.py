import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle

from simple_nn.NeuralNetwork import NeuralNetwork
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction


def one_hot(x: int, amount: int) -> np.ndarray:
    l = [0] * amount
    l[x] = 1
    return np.array(l)

iris = datasets.load_iris()

X = iris.data
Y = iris.target

Y = [one_hot(y, 3) for y in Y]

layer0 = Layer([4,10],1,ActivationFunction.LEAKY_RELU)
layer1 = Layer([10,10],1,ActivationFunction.LEAKY_RELU)
layer2 = Layer([10,3],1,ActivationFunction.SOFTMAX)
layers = [layer0, layer1,layer2]
nn = NeuralNetwork(layers)

# To remove order
X, Y = shuffle(X, Y, random_state=1)
nn.train(X,Y,10,.01,.9)
# Again with different order
X, Y = shuffle(X, Y, random_state=55)
nn.train(X,Y,10,.0005,.7)

pred, _ = nn.test(X,Y)

c = 0
for y, p in zip(Y, pred):
    predicted_class = np.argmax(p)
    true_class = np.argmax(y)

    if predicted_class == true_class:
        c += 1

# Calculate accuracy
accuracy = c / len(Y) * 100
print(f"\nAccuracy: {c}/{len(Y)} = {round(accuracy, 3)}%")






