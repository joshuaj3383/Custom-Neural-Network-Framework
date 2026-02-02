from typing import List, Union
import numpy as np
from simple_nn.ActivationFunction import ActivationFunction

class Layer:
    def __init__(
        self,
        weights: List[List[float] | List[int] | np.ndarray],
        biases: List[float] | float | np.ndarray,
        activationFunction: ActivationFunction = ActivationFunction.NONE,
    ):
        if not isinstance(activationFunction, ActivationFunction):
            raise TypeError(
                f"activationFunction must be of type ActivationFunction but is {type(activationFunction)}"
            )
        self.activationFunction = activationFunction

        # If shorthand: [num_inputs, num_neurons]
        if isinstance(weights, list) and len(weights) == 2 and all(isinstance(x, int) for x in weights):
            num_inputs, num_neurons = weights
            weights = self.activationFunction.initWeights(num_inputs, num_neurons)

        # Convert to NumPy arrays
        self.weights: np.ndarray = np.array(weights, dtype=np.float64)  # shape: (num_neurons, num_inputs)

        if isinstance(biases, (int, float)):
            biases = np.full((1,self.weights.shape[1]), biases, dtype=np.float64)
        self.biases: np.ndarray = np.array(biases, dtype=np.float64)  # shape: (num_neurons,)

        self.checkValid()

    def checkValid(self) -> bool:
        if self.weights.shape[1] != self.biases.shape[1]:
            raise ValueError(
                f"Weights and biases must have same number of neurons: {self.weights.shape[1]} != {self.biases.shape[1]}"
            )
        if self.weights.ndim != 2:
            raise ValueError(f"Weights must be 2D array, got shape {self.weights.shape}")
        return True

    def numNeurons(self) -> int:
        return self.weights.shape[1]

    def numInputs(self) -> int:
        return self.weights.shape[0]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Vectorized forward pass.
        inputs: shape (num_inputs,) or (batch_size, num_inputs)
        returns: shape (num_neurons,) or (batch_size, num_neurons)
        """

        z = inputs @ self.weights + self.biases
        return self.apply_activation(z)

    def apply_activation(self, z: np.ndarray) -> np.ndarray:
        if self.activationFunction == ActivationFunction.RELU:
            return np.maximum(0, z)
        elif self.activationFunction == ActivationFunction.LEAKY_RELU:
            return np.maximum(0.01 * z, z)
        elif self.activationFunction == ActivationFunction.SIGMOID:
            return 1 / (1 + np.exp(-z))
        elif self.activationFunction == ActivationFunction.TANH:
            return np.tanh(z)
        return z  # ActivationFunction.NONE

    def changeWeights(self, delta_weight: np.ndarray) -> None:
        if delta_weight.shape != self.weights.shape:
            raise ValueError(f"Weights have shape {self.weights.shape} while delta_weight has shape {delta_weight.shape}")

        self.weights += delta_weight

    def changeBiases(self, delta_bias: np.ndarray) -> None:
        if delta_bias.shape != self.biases.shape:
            raise ValueError(f"Biases have shape {self.biases.shape} while delta_bias has shape {delta_bias.shape}")
        self.biases += delta_bias

    def __str__(self):
        return f"Weights: {self.weights}, Biases: {self.biases}, Activation: {self.activationFunction}"

    def __repr__(self):
        return f"{self.weights}, {self.biases}, {self.activationFunction}"
