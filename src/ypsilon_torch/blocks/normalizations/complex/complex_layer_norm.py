from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from ypsilon_torch import FPDTypeIdx, get_complex_dtype

class ComplexLayerNorm(nn.Module):
    """LayerNorm for complex tensors, normalizing by magnitude statistics.

    Operates over the channel dimension. Computes mean and variance from
    ``|Z|`` (real-valued), then divides the complex ``Z`` by
    ``sqrt(var + eps)``. Applies learnable complex affine ``gamma * Z + beta``.

    Parameters
    ----------
    d_prime : int
        Channel dimension.
    eps : float
        Stability epsilon for variance. Default ``1e-5``.
    affine : bool
        If True, apply learnable ``gamma`` and ``beta``. Default True.

    Shape contract
    --------------
    Input  : ``(B, d_prime, H, W)`` complex.
    Output : same shape, complex.
    """

    def __init__(
        self,
        d_prime: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype_idx: FPDTypeIdx = 64
    ) -> None:
        super().__init__()
        self.d_prime: int = d_prime
        self.eps: float = eps
        self.affine: bool = affine

        if affine:
            self.gamma: Tensor = nn.Parameter(torch.empty(d_prime, dtype=get_complex_dtype(dtype_idx)))
            self.beta: Tensor = nn.Parameter(torch.empty(d_prime, dtype=get_complex_dtype(dtype_idx)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize affine parameters: gamma=1, beta=0 (both complex)."""
        if self.affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, Z: Tensor) -> Tensor:
        """Normalize and apply affine.

        Parameters
        ----------
        Z : Tensor
            Complex tensor of shape ``(B, d_prime, H, W)``.

        Returns
        -------
        Tensor
            Same shape, complex.
        """
        m: Tensor = Z.abs()  # real, (B, d_prime, H, W)
        mean: Tensor = m.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var: Tensor = ((m - mean) ** 2).mean(dim=1, keepdim=True)
        scale: Tensor = (var + self.eps).rsqrt()  # (B, 1, H, W) real
        Z_norm: Tensor = Z * scale  # complex * real -> complex
        if self.affine:
            return self.gamma.view(1, -1, 1, 1) * Z_norm + self.beta.view(1, -1, 1, 1)
        else:
            return Z_norm