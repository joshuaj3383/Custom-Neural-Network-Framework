import math

import numpy as np

from simple_nn.NeuralNetwork import NeuralNetwork
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction
import matplotlib.pyplot as plt

if __name__ == "__main__":
    layer0 = Layer([1,15],1,ActivationFunction.LEAKY_RELU)
    layer1 = Layer([15,15],1,ActivationFunction.LEAKY_RELU)
    layer2 = Layer([15,1],1,ActivationFunction.NONE)

    nn = NeuralNetwork([layer0,layer1,layer2], normalize_inputs=False)

    #print(len(nn.layers))

    X, Y = nn.generate_data(3000,(0,100),lambda x:math.log(x), .25)

    nn.train(X,Y,20,.0002,1)

    y_pred, _ = nn.test(X,Y)
    plt.scatter(X,Y,color="red",label="data",s=1)
    plt.scatter(X,y_pred,color="blue",label="prediction",s=5)

    plt.legend()
    plt.show()

