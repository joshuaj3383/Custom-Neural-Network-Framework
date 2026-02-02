from simple_nn.NeuralNetwork import NeuralNetwork as nn
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction

layer = Layer([1,1],1,ActivationFunction.NONE)

nn = nn([layer])

print("Generating 5000 samples between -100,100 for 50x-12.2")
X, Y = nn.generate_data(5000, (-100,100), lambda x: 50*x-12.2)

nn.train(X, Y, 10, 1e-4)

print(f"{round(nn.layers[0].weights[0][0],2)}x+{round(nn.layers[0].biases[0][0],2)}")
