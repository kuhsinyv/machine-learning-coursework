# -*- coding: utf-8 -*-
"""网络层."""
from abc import ABC
from typing import Literal, Optional

import numpy as np

from nn.parameter import Parameter
from nn.typing import Tensor
from nn import initializers


class Layer(ABC):  # pylint: disable=too-few-public-methods
    """抽象网络层."""
    parameters: list[Parameter]


class Linear2D(Layer):
    """2D 全连接层."""

    def __init__(self, in_features: int, out_features: int,
                 initializer: Literal["normal", "he_uniform"]) -> None:
        self.weight = Parameter(np.random.randn(out_features, in_features))
        self.bias = Parameter(np.random.randn(out_features))
        self.initialize(initializer)
        self.parameters = [self.weight, self.bias]
        self.inputs: Optional[Tensor] = None

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        """前向传播."""
        self.inputs = inputs  # (d_in, N)
        return self.weight.data @ inputs + np.expand_dims(self.bias.data,
                                                          axis=1)  # (d_out, N)

    def backward(self, grad_outputs: Tensor) -> Tensor:
        """反向传播."""
        if self.inputs is None:
            raise ValueError
        if self.weight.grad is not None:
            self.weight.grad += np.mean(
                np.expand_dims(grad_outputs.T, axis=2) *
                np.expand_dims(self.inputs.T, axis=1),
                axis=0
            )  # (N, d_out, 1) * (N, 1, d_in) -> (N, d_out, d_in) -> (d_out, d_in)
        if self.bias.grad is not None:
            self.bias.grad += grad_outputs.mean(
                axis=1)  # (d_out, N) -> (d_out, )

        self.inputs = None
        return self.weight.data.T @ grad_outputs  # (d_in, d_out) @ (d_out, N) -> (d_in, N)

    def initialize(self, initializer: Literal["normal", "he_uniform"]) -> None:
        """初始化参数."""
        match initializer:
            case "normal":
                self.weight.data = initializers.normal(self.weight.data.shape)
                self.bias.data = initializers.zeros(self.bias.data.shape)
            case "he_uniform":
                self.weight.data = initializers.he_uniform(
                    self.weight.data.shape)
                self.bias.data = initializers.zeros(self.bias.data.shape)


class ReLU(Layer):
    """ReLU 激活层."""
    inputs_stack: list[Tensor] = []  # 可复用对象

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        """前向传播."""
        self.inputs_stack.append(inputs)  # (d_in, N)
        return np.maximum(0.0, inputs)  # d_in = d_out, (d_out, N)

    def backward(self, grad_outputs: Tensor) -> Tensor:
        """反向传播."""
        if len(self.inputs_stack) == 0:
            raise ValueError
        grad_inputs = grad_outputs.copy()  # d_in = d_out, (d_in, N)
        grad_inputs[self.inputs_stack.pop() <= 0.0] = 0.0  # (d_in, N)
        return grad_inputs  # (d_in, N)


class Softmax(Layer):
    """数值稳定的 Softmax 层."""

    def __init__(self) -> None:
        self.outputs_stack: list[Tensor] = []

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        """forward function."""
        inputs_shift = inputs - np.max(inputs, axis=0,
                                       keepdims=True)  # (d_in, N)
        inputs_exp = np.exp(inputs_shift)  # (d_in, N)
        partition = inputs_exp.sum(axis=0, keepdims=True)  # (1, N)
        outputs = inputs_exp / partition  # (d_in, N)
        self.outputs_stack.append(outputs)
        return outputs  # d_in = d_out, (d_out, N)

    def backward(self, grad_outputs: Tensor) -> Tensor:
        """backward function."""
        if len(self.outputs_stack) == 0:
            raise ValueError
        outputs = self.outputs_stack.pop()  # (d_out, N)
        return (
            np.expand_dims(grad_outputs.T, axis=1)
            @ (-1.0 * np.einsum("ik, jk -> kij", outputs, outputs) +
               (np.expand_dims(outputs.T, axis=2) * np.eye(outputs.shape[0])))
        ).squeeze(
            axis=1
        ).T  # (N, 1, d_out) @ (N, d_out, d_out) -> (N, 1, d_out) -> (d_out, N) -> (d_in, N)
