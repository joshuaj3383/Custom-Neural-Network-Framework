from simple_nn.NeuralNetwork import NeuralNetwork
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction
import matplotlib.pyplot as plt


layer0 = Layer([1,10],.5,ActivationFunction.LEAKY_RELU)
layer1 = Layer([10,15],.5,ActivationFunction.LEAKY_RELU)
layer2 = Layer([15,10],1,ActivationFunction.LEAKY_RELU)
layer3 = Layer([10,1],1.5,ActivationFunction.NONE)


layers = [layer0, layer1, layer2, layer3]

nn = NeuralNetwork(layers)

print("Generating 5000 samples for x in -5,5 with x^3+3x^2-1")
X, Y = nn.generate_data(1000, (-5,5), lambda x:x*x*x+3*x*x-1, 5)

nn.train(X, Y, 20, .0001, .9)

results, loss = nn.test(X, Y)

plt.figure()
plt.scatter(X,Y,color="red")
plt.title("Data")
plt.show()
plt.figure()
plt.scatter(X,results,color="blue")
plt.title("Predictions")
plt.show()


