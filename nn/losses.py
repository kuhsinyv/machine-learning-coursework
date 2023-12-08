# -*- coding: utf-8 -*-
"""损失函数."""
from typing import Literal, Optional

import numpy as np
from nn.typing import Tensor


class CrossEntropyLoss:
    """交叉熵损失函数."""

    def __init__(self, reduction: Literal["mean", "sum"]) -> None:
        self.reduction = reduction

        self.inputs: Optional[Tensor] = None
        self.targets: Optional[Tensor] = None

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return self.forward(inputs, targets)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """forward function."""
        self.inputs = inputs  # (d_out, N)
        self.targets = targets  # (d_out, N)
        outputs = -1.0 * (targets * np.log(inputs)).sum(axis=1)  # (N, )
        match self.reduction:
            case "mean":
                return np.mean(outputs)  # ()
            case "sum":
                return np.sum(outputs)  # ()

        raise ValueError

    def backward(self) -> Tensor:
        """backward function."""
        if self.inputs is None or self.targets is None:
            raise ValueError
        grad_inputs = -1.0 * (self.targets / self.inputs)
        self.inputs = None
        self.targets = None
        return grad_inputs  # (d_in, N)
