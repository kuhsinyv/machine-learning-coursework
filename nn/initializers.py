# -*- coding: utf-8 -*-
"""权重初始化方法."""
import numpy as np

from nn.typing import DType, Tensor, Shape


def zeros(shape: Shape, dtype: type = DType) -> Tensor:
    """全 0 初始化."""
    return np.zeros(shape=shape, dtype=dtype)


def normal(shape: Shape,
           dtype: type = DType,
           mean: float = 0.0,
           std: float = 1.0) -> Tensor:
    """normal 初始化."""
    return np.random.normal(loc=mean, scale=std, size=shape).astype(dtype)


def he_uniform(shape: tuple[int, ...], dtype: type = DType) -> Tensor:
    """何凯明 uniform 初始化."""
    limit = np.sqrt(6.0 / shape[1])
    return np.random.uniform(low=-limit, high=limit, size=shape).astype(dtype)
