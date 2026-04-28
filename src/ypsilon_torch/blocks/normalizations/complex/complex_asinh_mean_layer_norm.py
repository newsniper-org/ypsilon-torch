from __future__ import annotations

import torch
from torch import Tensor, nn
from ypsilon_torch import FPDTypeIdx, get_complex_dtype, get_float_dtype


def _make_view_shape(ndim: int, dim: int) -> list[int]:
    shape = [1] * ndim
    shape[dim] = -1
    return shape


class ComplexAsinhMeanLayerNorm(nn.Module):
    r"""Magnitude-based arsinh layer normalization for complex tensors.

    Applies arsinh to the **magnitude** of the complex input, computes
    mean/variance in the transformed space, normalizes, and rescales the
    complex tensor while preserving phase.

    .. math::

        m_j      &= |z_j|

        u_j      &= \operatorname{arsinh}(m_j / c)

        \hat{u}_j &= \frac{u_j - \mathbb{E}(U)}
                          {\sqrt{\mathbb{V}(U) + \epsilon}}

        \mathrm{out}_j &= \gamma_j\,
                          z_j \cdot \frac{\hat{u}_j}{m_j + \epsilon}
                          + \beta_j

    Parameters
    ----------
    normalized_shape : int
        Size of the dimension to normalize.
    c_init : float
        Initial value for the learnable arsinh scale. Default ``1.0``.
    eps : float
        Stability epsilon. Default ``1e-5``.
    dim : int
        Dimension to normalize over. Default ``-1``.
    affine : bool
        If True, apply learnable complex ``gamma`` and ``beta``.
    dtype_idx : FPDTypeIdx
        Floating-point precision index. Default ``64``.
    """

    def __init__(
        self,
        normalized_shape: int,
        c_init: float = 1.0,
        eps: float = 1e-5,
        dim: int = -1,
        affine: bool = True,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        self.normalized_shape: int = normalized_shape
        self.eps: float = eps
        self.dim: int = dim
        self.affine: bool = affine

        float_dtype = get_float_dtype(dtype_idx)
        self.c: nn.Parameter = nn.Parameter(
            torch.tensor(c_init, dtype=float_dtype)
        )

        complex_dtype = get_complex_dtype(dtype_idx)
        if affine:
            self.gamma: nn.Parameter = nn.Parameter(
                torch.empty(normalized_shape, dtype=complex_dtype)
            )
            self.beta: nn.Parameter = nn.Parameter(
                torch.empty(normalized_shape, dtype=complex_dtype)
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, z: Tensor) -> Tensor:
        m = z.abs()                                          # real magnitudes
        u = torch.asinh(m / self.c)

        mean = u.mean(dim=self.dim, keepdim=True)
        var = ((u - mean).pow(2)).mean(dim=self.dim, keepdim=True)
        u_norm = (u - mean) / torch.sqrt(var + self.eps)     # real normalized

        # Rescale complex tensor: preserve phase, apply normalised magnitude
        z_norm = z * (u_norm / (m + self.eps))

        if self.affine:
            view = _make_view_shape(z.ndim, self.dim)
            return self.gamma.view(*view) * z_norm + self.beta.view(*view)
        return z_norm
