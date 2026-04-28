import math
import torch
import torch.nn as nn

try:
    # Python 3.12+
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override

from ypsilon_torch import FPDTypeIdx
from ypsilon_torch.blocks.activations import BiasedComplexActivationFunctionBase


class StableComplexCardioid(BiasedComplexActivationFunctionBase):
    """
    Stable implementation of ComplexCardioid
    """
    @override
    def __init__(self, features: int, eps: float = 1e-6, dtype_idx: FPDTypeIdx = 64):
        super().__init__(features, eps, dtype_idx)
    
    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D1, D2, ..., C) complex

        # 1. Learnable mu 더하기 & ReLU 적용
        # m shape: (C,) -> (1, 1, 1, ..., C) broadcasting
        view_shape = [1] * (z.ndim - 1) + [-1]
        mu = self.bias.view(*view_shape)

        # 2. Rescale z (위상 유지)
        arg: torch.Tensor = torch.atan2(z.imag, z.real) - mu
        coeff: torch.Tensor = (torch.cos(arg) + 1) / 2
        return (coeff + self.eps) * z