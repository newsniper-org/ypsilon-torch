from __future__ import annotations

import torch
from torch import Tensor, nn
from ypsilon_torch import FPDTypeIdx, get_float_dtype


def _make_view_shape(ndim: int, dim: int) -> list[int]:
    """Build a broadcast shape with -1 at *dim* and 1 elsewhere."""
    shape = [1] * ndim
    shape[dim] = -1
    return shape


class AsinhMeanLayerNorm(nn.Module):
    r"""Layer normalization in arsinh-transformed space.

    .. math::

        z_j = \operatorname{arsinh}(x_j / c)

        \operatorname{AsinhMeanLayerNorm}(X)_j
            = \gamma_j \frac{z_j - \mathbb{E}(Z)}
                             {\sqrt{\mathbb{V}(Z) + \epsilon}}
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
        If True, apply learnable ``gamma`` and ``beta``. Default True.
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

        if affine:
            self.gamma: nn.Parameter = nn.Parameter(
                torch.empty(normalized_shape, dtype=float_dtype)
            )
            self.beta: nn.Parameter = nn.Parameter(
                torch.empty(normalized_shape, dtype=float_dtype)
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, x: Tensor) -> Tensor:
        z = torch.asinh(x / self.c)

        mean = z.mean(dim=self.dim, keepdim=True)
        var = ((z - mean).pow(2)).mean(dim=self.dim, keepdim=True)
        z_norm = (z - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            view = _make_view_shape(x.ndim, self.dim)
            return self.gamma.view(*view) * z_norm + self.beta.view(*view)
        return z_norm
