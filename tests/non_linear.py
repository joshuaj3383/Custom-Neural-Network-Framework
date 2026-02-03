from simple_nn.NeuralNetwork import NeuralNetwork
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction
import matplotlib.pyplot as plt

print("Info===============================")
print("Creating Neural Network for layers of size [1,25][25,25][25,1]")
print("This corresponds to three layers with 1 input, 25 hidden neurons, 25 hidden neurons, one output layer")
print("The hidden neurons are leak_relu to remove linearity and to avoid dead neurons")
print("The output layer have no activation function to allow for negative values")

#Create Layers
layer0 = Layer([1,25],-1,ActivationFunction.LEAKY_RELU)
layer1 = Layer([25,25],.5,ActivationFunction.LEAKY_RELU)
layer2 = Layer([25,1],1,ActivationFunction.NONE)
layers = [layer0, layer1, layer2]
#Create NN
nn = NeuralNetwork(layers)

#Train
print("=====================================")
print("Generating 1000 samples for x in -7,5 with .25x^3+1.5x^2-3 with noist_std=2")
print("These numbers were chosen to provide an interesting graph with 3 roots and 2 minima/maxima")
X, Y = nn.generate_data(5000, (-7,5), lambda x:.25*x*x*x+1.5*x*x-3, 2)
print("\nTraining=============================")
nn.train(X, Y, 7, .0003, .9)

#Test
x = -7
in_nn = []
out_nn = []
while x < 5:
    in_nn.append(round(x,3))
    out_nn.append(round(nn.predict([x]).tolist()[0][0],3))
    x += 0.05

#Plot
print("\nPlotting===================================")
plt.figure()
plt.scatter(X,Y,color="red",s=3)
plt.title("Data")
plt.show()
plt.figure()
plt.scatter(in_nn,out_nn,color="blue",s=10)
plt.title("Predictions")
plt.show()
print("Done")

#Check zeros, minima, and maxima
zeros = []
minima = []
maxima = []
#Loops through inputs, avoid first and last
i = 1
while i < len(in_nn) - 1:
    prv = out_nn[i-1]
    cur = out_nn[i]
    nxt = out_nn[i+1]

    x = in_nn[i]

    # If there is a change in sign it is a root
    if prv < 0 < nxt or prv > 0 > nxt:
        zeros.append((x,cur))
        i += 1 #avoid doubling roots


    # If cur is less than both its neighbors it is a minima
    if prv > cur and nxt > cur:
        minima.append((x, cur))

    # If cur is greater than both its neighbors it is a maxima
    if prv < cur and nxt < cur:
        maxima.append((x,cur))

    i+=1



print("\nAnalysis==========================================================")
print(f"Expected zeros:   [(-5.62,0), (-1.66,0), (+1.28,0)]")
print(f"Predicted Zeros:  {zeros}")
print(f"Expected minima:  [(0,-3)]")
print(f"Predicted minima: {minima}")
print(f"Expected maxima:  (-4,5)")
print(f"Predicted maxima: {maxima}")


#print([(x,y) for x,y in zip(in_nn, out_nn)])









