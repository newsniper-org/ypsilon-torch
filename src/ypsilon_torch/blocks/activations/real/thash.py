import torch
import torch.nn as nn

from ypsilon_torch.blocks.activations import RealActivationFunctionBase
from ypsilon_torch.functional import thash

class ThASh(RealActivationFunctionBase):
    r"""
    ThASh(TanhArSinh) activation.

    Definition:
        ThASh(x) = tanh(asinh(x)) = x / sqrt(1 + x^2)

    This is designed as a drop-in replacement for built-in PyTorch
    activations such as nn.Tanh() / nn.Softsign() in contexts where:
      - domain  = R
      - codomain = R
      - image   = (-1, 1)

    Shape:
      - Input:  (*)
      - Output: (*), same shape as input
    """

    __constants__ = []

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return thash(input)

    def extra_repr(self) -> str:
        return ""