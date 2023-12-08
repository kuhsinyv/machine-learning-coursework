# -*- coding: utf-8 -*-
"""网络参数类型."""
import numpy as np

from nn.typing import DType, Tensor


class Parameter:
    """网络参数."""

    def __init__(self,
                 data: Tensor,
                 requires_grad: bool = True,
                 copy: bool = True) -> None:
        self.data = np.array(data, DType, copy=copy)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad

    def zero_grad(self) -> None:
        """梯度清零."""
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

    def apply_update(self, data: Tensor) -> None:
        """更新参数."""
        self.data = data
