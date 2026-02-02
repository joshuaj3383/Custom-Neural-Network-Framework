from enum import Enum
import numpy as np

class ActivationFunction(Enum):
    NONE = 0
    RELU = 2
    LEAKY_RELU = 4
    SIGMOID = 1
    TANH = 3
    SOFTMAX = 5  # Added Softmax

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
            # Softmax: convert logits to probabilities
            exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
            return exp_values / np.sum(exp_values, axis=-1, keepdims=True)  # Normalize
        return x

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
            # Softmax doesn't have a simple derivative, handle it in loss function
            # Cross-entropy with softmax is commonly used, which combines them
            raise NotImplementedError("Softmax derivative is handled in the loss function")
        return x

    def initWeights(self, numInputs: int, numNeurons: int) -> np.ndarray:
        scale = 0.01
        if self in {ActivationFunction.SIGMOID, ActivationFunction.TANH}:
            scale = (2 / (numInputs + numNeurons)) ** 0.5
        if self in {ActivationFunction.RELU, ActivationFunction.LEAKY_RELU}:
            scale = (2 / numInputs) ** 0.5

        return np.random.normal(0, scale, [numInputs, numNeurons])
