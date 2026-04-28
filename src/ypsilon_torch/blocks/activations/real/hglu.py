import torch
import torch.nn as nn

from ypsilon_torch.blocks.activations import RealActivationFunctionBase
from ypsilon_torch.functional import hglu

class HGLU(RealActivationBase):
    r"""
    HGLU_k(Hyperbolic Gain Linear Unit with positive hyperparameter k) activation

    Definition:
        HGLU_k(x) = (x + sqrt(k + x^2)) / 2

    This is designed as a drop-in replacement for built-in PyTorch
    activations such as nn.Tanh() / nn.Softsign() in contexts where:
      - domain  = R
      - codomain = R
      - image   = (0, +inf)

    Shape:
      - Input:  (*)
      - Output: (*), same shape as input
    """

    __constants__ = []

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k: float = k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return hglu(input, self.k)

    def extra_repr(self) -> str:
        return ""