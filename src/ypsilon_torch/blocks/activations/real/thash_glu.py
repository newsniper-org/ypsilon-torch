import torch
import torch.nn as nn

from ypsilon_torch.blocks.activations import RealActivationFunctionBase
from ypsilon_torch.functional import thash_glu

class ThAShGLU(RealActivationFunctionBase):
    r"""
    ThAShGLU (ThASh-Gated Linear Unit).

    Definition:
        split x into (a, b) along ``dim``;  ThAShGLU(x) = a ⊙ ThASh(b)
        where ThASh(b) = b / sqrt(1 + b^2) is bounded to (-1, 1).

    A Gated Linear Unit variant whose gate saturates more gently than a
    sigmoid gate. As with ``nn.GLU``, the output halves the gated dimension.

      - domain  = R
      - codomain = R

    Shape:
      - Input:  (*, 2D, *)  along ``dim``
      - Output: (*, D, *)   (``dim`` halved)
    """

    __constants__ = ["dim"]

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim: int = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return thash_glu(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
