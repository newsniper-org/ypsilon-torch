import torch
import torch.nn as nn

try:
    # Python 3.12+
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override

from ypsilon_torch.blocks.activations import ComplexActivationFunctionBase
from ypsilon_torch.functional import complex_thash


class ComplexThASh(ComplexActivationFunctionBase):
    r"""
    ComplexThASh: magnitude-ThASh with phase preserved.

    Definition:
        f(z) = (z / |z|) · ThASh(|z|) = z / sqrt(1 + |z|^2)

    Squashes the modulus into (0, 1) while leaving arg(z) untouched — the
    complex-domain analogue of the real :class:`ThASh`. Parameter-free and
    shape-preserving. The singularity at z = 0 is removed analytically
    (the formula never divides by |z|).
    """

    __constants__ = ["eps"]

    @override
    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps: float = eps

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return complex_thash(z, eps=self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
