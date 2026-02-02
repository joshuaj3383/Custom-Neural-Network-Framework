from sklearn import datasets

from simple_nn.NeuralNetwork import NeuralNetwork
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data
Y = iris.target

layer0 = Layer([4,5],-1,ActivationFunction.LEAKY_RELU)
layer1 = Layer([5,5],-.5,ActivationFunction.LEAKY_RELU)
layer2 = Layer([5,3],.5,ActivationFunction.SOFTMAX)


layers = [layer0, layer1, layer2]

nn = NeuralNetwork(layers)

nn.train(X,Y,5,0.01)






