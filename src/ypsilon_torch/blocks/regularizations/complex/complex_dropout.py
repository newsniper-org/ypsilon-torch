import torch
import torch.nn as nn

from ypsilon_torch.functional import complex_dropout

class ComplexDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p: float = p

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return complex_dropout(z,self.p,self.training)