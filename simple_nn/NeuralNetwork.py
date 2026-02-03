from typing import List, Callable
import numpy as np
import ast
from simple_nn.Layer import Layer
from simple_nn.ActivationFunction import ActivationFunction


class NeuralNetwork:
    def __init__(self, layers: List[Layer] | str, normalize_inputs: bool = False):
        if not layers:
            raise ValueError("NeuralNetwork must have at least one layer.")

        if isinstance(layers, str):
            list_repr = ast.literal_eval(layers)
            layers = []
            for layer in list_repr:
                if len(layer) != 3:
                    raise ValueError(f"Expected 3 layers, got {len(layer)} in layer {layer}")
                layers.append(Layer(layer[0], layer[1], layer[2]))

        else:
            for layer in layers:
                if not isinstance(layer, Layer):
                    raise ValueError(f"Expected a Layer object, got {type(layer)}")

        self.layers: List[Layer] = layers
        self.normalize_inputs = normalize_inputs
        #We force softmax to be paired with CEL for practicality and also math
        self.loss_func = self.cel if layers[-1].activationFunction == ActivationFunction.SOFTMAX else self.mse
        self._check_dimensions()

    def _check_dimensions(self):
        """Ensures the layers dimensions are correct so the outputs of one can be passed as the inputs for another"""
        for i in range(len(self.layers) - 1):
            assert self.layers[i].numNeurons() == self.layers[i + 1].numInputs(), \
                f"Layer {i}: {self.layers[i].numNeurons()} neurons != Layer {i + 1}: {self.layers[i + 1].numInputs()} inputs"

    def normalize_input(self, X: np.ndarray) -> np.ndarray:
        """Normalize the inputs if required"""
        if self.normalize_inputs:
            return X / np.max(np.abs(X), axis=0)
        else:
            return X

    def forward(self, X: np.ndarray | list) -> List[np.ndarray]:
        """Forward pass returning outputs of all layers"""

        if isinstance(X, list):
            X = np.array(X).reshape(1, -1)

        X = self.normalize_input(X)
        outputs = [X]

        for i, layer in enumerate(self.layers):
            #print(f"Layer {i} forward pass:")
            X = layer.forward(X)
            outputs.append(X)

        return outputs

    def backprop(self, X: np.ndarray, Y: np.ndarray, lr: float):
        """Perform one backpropagation step"""

        # Make sure that the data is at least 2d for dimsions sake
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Setup
        activations = self.forward(X)
        error = [None] * len(self.layers)

        # Output layer error
        output = activations[-1]

        # Make sure shapes line up bc of brocasting
        # eg. (1,3) - (1,1) will fly by (thx iris)
        if output.shape != Y.shape:
            raise ValueError(f"Network output shape: {output.shape} != data shape: {Y.shape}")

        # We force softmax to be paired with CEL for practicality and also math
        if self.layers[-1].activationFunction == ActivationFunction.SOFTMAX:
            error[-1] = output - Y
        else:
            #print(f"sizes: ({output.shape}-{Y.shape}) * {self.layers[-1].activationFunction.calcDerivative(output).shape}")
            error[-1] = (output - Y) * self.layers[-1].activationFunction.calcDerivative(output)
        # Backpropagate through hidden layers
        for i in reversed(range(len(self.layers) - 1)):
            W_next = self.layers[i + 1].weights
            error[i] = (error[i + 1] @ W_next.T) * self.layers[i].activationFunction.calcDerivative(activations[i + 1])

        # Update weights and biases
        for i, layer in enumerate(self.layers):
            #print("-----------------------------------------")
            X_input = activations[i].reshape(1, -1)
            #print(f"x_input[{i}]: {X_input.shape}")
            delta_W = np.outer(X_input, error[i])
            #print(f"error[{i}]: {error[i].shape}")
            #print(f"delta_W for layer {i}: {delta_W.shape}")

            layer.changeWeights(-lr * delta_W)
            layer.changeBiases(-lr * error[i])

        return error

    def train(self, X: np.ndarray | list, Y: np.ndarray | list, epochs: int, lr: float, decay: float = 1):
        for epoch in range(epochs):
            net_loss = 0
            for x, y in zip(X, Y):
                #print(x,y)
                self.backprop(x, y, lr)
                loss = self.loss_func(self.forward(x)[-1], y)
                #print(self.forward(x)[-1],y,loss)
                net_loss += loss

            avg_loss = net_loss / len(X)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")
            lr *= decay

    def test(self, X: np.ndarray, Y: np.ndarray) -> tuple[List[float], List[float]]:
        predictions = []
        losses = []

        for x, y in zip(X, Y):
            # Force dimensions for 1d
            x = np.atleast_2d(x)
            y = np.atleast_2d(y)

            pred = self.predict(x)
            predictions.append(pred)
            losses.append(self.loss_func(pred, y))

        return predictions, losses

    def predict(self, X: np.ndarray | list) -> np.ndarray:
        """Gets the final result from an input"""
        return self.forward(X)[-1]

    @staticmethod
    def mse(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """Mean squared error"""
        return float(np.mean((Y_pred - Y_true) ** 2))

    @staticmethod
    def cel(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """Cross entropy loss"""
        epsilon = 1e-5
        #print(Y_pred)
        return float(-1 * np.sum(Y_true * np.log(Y_pred + epsilon)))  # Cross-entropy

    @staticmethod
    def generate_data(num_samples: int, x_range: tuple, func: Callable[[float], float], noise_std: float = 0):
        X = np.random.uniform(x_range[0], x_range[1], size=(num_samples,))
        Y = np.array([func(x) for x in X])
        if noise_std > 0:
            Y += np.random.normal(0, noise_std, size=Y.shape)
        return X, Y

    def __str__(self) -> str:
        string = ""
        for i, layer in enumerate(self.layers):
            string += f"\nLayer {i}---------------\n"
            string += layer.__str__()

        return string

    def __repr__(self) -> str:
        string = "["
        for layer in self.layers:
            string += layer.__repr__() + ", "
        string += "]"
        return string