from __future__ import annotations

import torch
from torch import Tensor, nn
from ypsilon_torch import FPDTypeIdx, get_complex_dtype


def _make_view_shape(ndim: int, dim: int) -> list[int]:
    shape = [1] * ndim
    shape[dim] = -1
    return shape


class ComplexRobustLayerNorm(nn.Module):
    r"""Robust layer normalization for complex-valued tensors.

    Uses the Weiszfeld algorithm (geometric median in the complex plane)
    for centering and MAD of magnitudes for scaling.  Phase is preserved.

    .. math::

        \mu_\phi = \arg\min_{y \in \mathbb{C}}
                   \sum_j |z_j - y|^\phi

        s_\phi   = \mathrm{med}_j\, |z_j - \mu_\phi|

        \mathrm{ComplexRobustLayerNorm}(Z)_j
            = \gamma_j \frac{z_j - \mu_\phi}{s_\phi + \epsilon} + \beta_j

    Parameters
    ----------
    normalized_shape : int
        Size of the dimension to normalize.
    phi : float
        Exponent for the location estimator (``1 < phi < 2``).
    eps : float
        Stability epsilon. Default ``1e-5``.
    dim : int
        Dimension to normalize over. Default ``-1``.
    affine : bool
        If True, apply learnable complex ``gamma`` and ``beta``.
    dtype_idx : FPDTypeIdx
        Floating-point precision index. Default ``64``.
    max_iter : int
        Maximum Weiszfeld iterations. Default ``10``.
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
        # --- Weiszfeld location in C ---
        mu = z.mean(dim=self.dim).unsqueeze(self.dim)

        with torch.no_grad():
            for _ in range(self.max_iter):
                residual = (z - mu).abs().clamp(min=1e-12)  # real magnitudes
                w = residual.pow(self.phi - 2)
                mu_new = (w * z).sum(dim=self.dim, keepdim=True) / w.sum(
                    dim=self.dim, keepdim=True
                )
                if (mu_new - mu).abs().max() < 1e-6:
                    break
                mu = mu_new

        # --- MAD scale (real-valued) ---
        s = torch.median(
            (z - mu).abs(), dim=self.dim
        ).values.unsqueeze(self.dim)

        # --- normalize ---
        z_norm = (z - mu) / (s + self.eps)

        if self.affine:
            view = _make_view_shape(z.ndim, self.dim)
            return self.gamma.view(*view) * z_norm + self.beta.view(*view)
        return z_norm
