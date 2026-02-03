import ast

from simple_nn.NeuralNetwork import NeuralNetwork
from simple_nn.Layer import Layer


layer0 = Layer([1,3],1,)
layer1 = Layer([3,2],1)

nn = NeuralNetwork([layer0,layer1])



print(nn.__repr__())

s = "[[[[-0.005875191111360565, 0.011821665986938156, 0.004804463470497995]], [[1.0, 1.0, 1.0]], 0], [[[0.0023667438882052673, 0.0036003154203368055], [-0.017163067371453674, 0.007159528684803832], [-0.023598434525129212, 0.008122708836355444]], [[1.0, 1.0]], 0], ]"
l = ast.literal_eval(s)

nn_c = NeuralNetwork(s)

print(nn_c)




