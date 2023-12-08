# -*- coding: utf-8 -*-
"""优化器."""

from nn.parameter import Parameter

class SGD:
    """Implements stochastic gradient descent."""

    def __init__(self, parameters: list[Parameter], lr: float = 0.001) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """清零梯度."""
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()

    def step(self):
        """更新参数."""
        for param in self.parameters:
            if param.grad is not None:
                param.apply_update(param.data - self.lr * param.grad)
