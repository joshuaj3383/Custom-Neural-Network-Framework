from enum import Enum
import numpy as np

class ActivationFunction(Enum):
    NONE = 0
    RELU = 1
    LEAKY_RELU = 2
    SIGMOID = 3
    TANH = 4
    SOFTMAX = 5

    def calcActivationFunction(self, x: np.ndarray) -> np.ndarray:
        if self == ActivationFunction.NONE:
            return x
        elif self == ActivationFunction.RELU:
            return np.maximum(0, x)
        elif self == ActivationFunction.LEAKY_RELU:
            return np.maximum(0.01 * x, x)
        elif self == ActivationFunction.SIGMOID:
            return 1 / (1 + np.exp(-x))
        elif self == ActivationFunction.TANH:
            return np.tanh(x)
        elif self == ActivationFunction.SOFTMAX:
            return np.exp(x) / np.sum(np.exp(x))
        else:
            raise ValueError("Unknown activation function")

    def calcDerivative(self, x: np.ndarray) -> np.ndarray:
        if self == ActivationFunction.NONE:
            return np.ones_like(x)
        elif self == ActivationFunction.RELU:
            return np.where(x <= 0, 0, 1)
        elif self == ActivationFunction.LEAKY_RELU:
            return np.where(x <= 0, 0.01, 1)
        elif self == ActivationFunction.SIGMOID:
            return x * (1 - x)
        elif self == ActivationFunction.TANH:
            return 1 - np.square(x)
        elif self == ActivationFunction.SOFTMAX:
            raise ValueError("Softmax derivative should not be called as it is implemented in bpp with CEL")
        else:
            raise ValueError("Unknown activation function")

    def initWeights(self, numInputs: int, numNeurons: int) -> np.ndarray:
        scale = 0.01
        if self in {ActivationFunction.SIGMOID, ActivationFunction.TANH, ActivationFunction.SOFTMAX}:
            scale = (2 / (numInputs + numNeurons)) ** 0.5
        if self in {ActivationFunction.RELU, ActivationFunction.LEAKY_RELU}:
            scale = (2 / numInputs) ** 0.5

        return np.random.normal(0, scale, [numInputs, numNeurons])
