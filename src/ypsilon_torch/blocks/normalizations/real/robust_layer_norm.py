from __future__ import annotations

import torch
from torch import Tensor, nn
from ypsilon_torch import FPDTypeIdx, get_float_dtype


def _make_view_shape(ndim: int, dim: int) -> list[int]:
    """Build a broadcast shape with -1 at *dim* and 1 elsewhere."""
    shape = [1] * ndim
    shape[dim] = -1
    return shape


class RobustLayerNorm(nn.Module):
    r"""Layer normalization using an L_phi location estimator and MAD scale.

    .. math::

        \mu_\phi = \arg\min_y \sum_j |x_j - y|^\phi

        s_\phi  = \mathrm{med}_j\, |x_j - \mu_\phi|

        \mathrm{RobustLayerNorm}(x)_j
            = \gamma_j \frac{x_j - \mu_\phi}{s_\phi + \epsilon} + \beta_j

    Parameters
    ----------
    normalized_shape : int
        Size of the dimension to normalize.
    phi : float
        Exponent for the location estimator (``1 < phi < 2``).
        Recommended range: ``1.2 <= phi < 2``.
    eps : float
        Stability epsilon. Default ``1e-5``.
    dim : int
        Dimension to normalize over. Default ``-1``.
    affine : bool
        If True, apply learnable ``gamma`` and ``beta``. Default True.
    dtype_idx : FPDTypeIdx
        Floating-point precision index. Default ``64``.
    max_iter : int
        Maximum IRLS iterations. Default ``10``.
    """

    def __init__(
        self,
        normalized_shape: int,
        phi: float = 1.5,
        eps: float = 1e-5,
        dim: int = -1,
        affine: bool = True,
        dtype_idx: FPDTypeIdx = 64,
        max_iter: int = 10,
    ) -> None:
        super().__init__()
        if not (1.0 < phi < 2.0):
            raise ValueError(f"phi must be in (1, 2), got {phi}")

        self.normalized_shape: int = normalized_shape
        self.phi: float = phi
        self.eps: float = eps
        self.dim: int = dim
        self.affine: bool = affine
        self.max_iter: int = max_iter

        float_dtype = get_float_dtype(dtype_idx)
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
        # --- L_phi location (IRLS) ---
        mu = torch.median(x, dim=self.dim).values.unsqueeze(self.dim)

        with torch.no_grad():
            for _ in range(self.max_iter):
                residual = (x - mu).abs().clamp(min=1e-12)
                w = residual.pow(self.phi - 2)
                mu_new = (w * x).sum(dim=self.dim, keepdim=True) / w.sum(
                    dim=self.dim, keepdim=True
                )
                if (mu_new - mu).abs().max() < 1e-6:
                    break
                mu = mu_new

        # --- MAD scale ---
        s = torch.median((x - mu).abs(), dim=self.dim).values.unsqueeze(self.dim)

        # --- normalize ---
        x_norm = (x - mu) / (s + self.eps)

        if self.affine:
            view = _make_view_shape(x.ndim, self.dim)
            return self.gamma.view(*view) * x_norm + self.beta.view(*view)
        return x_norm
