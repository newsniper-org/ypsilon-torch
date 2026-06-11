import torch
import torch.nn as nn

try:
    # Python 3.12+
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override

from ypsilon_torch.blocks.activations import ComplexActivationFunctionBase
from ypsilon_torch.functional import zrelu


class ZReLU(ComplexActivationFunctionBase):
    r"""
    zReLU (Guberman, 2016).

    Definition:
        f(z) = z   if Re(z) > 0 and Im(z) > 0
             = 0   otherwise

    Passes a complex value through only when it lies in the (open) first
    quadrant of the complex plane. Parameter-free and shape-preserving.
    """

    __constants__ = []

    @override
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return zrelu(z)

    def extra_repr(self) -> str:
        return ""
