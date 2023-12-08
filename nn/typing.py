# -*- coding: utf-8 -*-
"""数据类型."""
from typing import TypeAlias

import numpy as np
from numpy import typing as npt

DType: TypeAlias = np.float64
Tensor: TypeAlias = npt.NDArray[DType]
Shape: TypeAlias = tuple[int, ...]
