# -*- coding: utf-8 -*-
"""网络."""
from typing import Literal

from nn.layers import Linear2D, ReLU, Softmax
from nn.typing import Tensor


class DNN:
    """深度神经网络."""

    def __init__(self, layers: list[int],
                 initializer: Literal["normal", "he_uniform"]) -> None:
        self.layers = [
            Linear2D(layers[i], layers[i + 1], initializer)
            for i in range(len(layers) - 1)
        ]
        self.relu = ReLU()
        self.softmax = Softmax()
        self.parameters = []
        for layer in self.layers:
            self.parameters += layer.parameters

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        """前向传播."""
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if i != len(self.layers) - 1:
                inputs = self.relu(inputs)
        return self.softmax(inputs)

    def backward(self, grad_outputs: Tensor) -> Tensor:
        """反向传播."""
        grad_outputs = self.softmax.backward(grad_outputs)
        for i in range(len(self.layers) - 1, -1, -1):
            grad_outputs = self.layers[i].backward(grad_outputs)
            if i != 0:
                grad_outputs = self.relu.backward(grad_outputs)
        return grad_outputs
