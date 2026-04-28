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
import torch.nn.functional as F

class StableModReLU(BiasedComplexActivationFunctionBase):
    """
    Stable implementation of ModReLU.
    z=0에서의 특이점을 해결하기 위해 epsilon smoothing을 사용하여 크기를 계산합니다.
    """
    @override
    def __init__(self, features: int, eps: float = 1e-6, dtype_idx: FPDTypeIdx = 64):
        super().__init__(features, eps, dtype_idx)

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D1, D2, ..., C) complex
        
        # 1. Smoothed magnitude 계산
        mag = torch.sqrt(z.real.pow(2) + z.imag.pow(2) + self.eps)
        
        # 2. Learnable bias 더하기 & ReLU 적용
        # bias shape: (C,) -> (1, 1, 1, ..., C) broadcasting
        view_shape = [1] * (z.ndim - 1) + [-1]
        b = self.bias.view(*view_shape)
        act_mag = F.relu(mag + b)
        
        # 3. Rescale z (위상 유지)
        return z * (act_mag / mag)