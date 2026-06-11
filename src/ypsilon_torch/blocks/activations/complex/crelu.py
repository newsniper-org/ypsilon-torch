import torch
import torch.nn as nn

try:
    # Python 3.12+
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override

from ypsilon_torch.blocks.activations import ComplexActivationFunctionBase
from ypsilon_torch.functional import crelu


class CReLU(ComplexActivationFunctionBase):
    r"""
    CReLU (Trabelsi et al., 2018).

    Definition:
        f(z) = ReLU(Re(z)) + i · ReLU(Im(z))

    Applies ReLU independently to the real and imaginary parts. Parameter-free
    and shape-preserving.
    """

    __constants__ = []

    @override
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return crelu(z)

    def extra_repr(self) -> str:
        return ""
