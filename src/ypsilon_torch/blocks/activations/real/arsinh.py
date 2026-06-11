import torch
import torch.nn as nn

from ypsilon_torch.blocks.activations import RealActivationFunctionBase
from ypsilon_torch.functional import arsinh

class ArSinh(RealActivationFunctionBase):
    r"""
    ArSinh activation.

    Definition:
        ArSinh(x) = asinh(x) = log(x + sqrt(1 + x^2))

    Odd, unbounded, log-compressive: behaves like the identity near the
    origin and like ``sign(x) * log(2|x|)`` in the tails. Useful as a gentle
    replacement for the identity on heavy-tailed pre-activations.

      - domain  = R
      - codomain = R
      - image   = R

    Shape:
      - Input:  (*)
      - Output: (*), same shape as input
    """

    __constants__ = []

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return arsinh(input)

    def extra_repr(self) -> str:
        return ""
