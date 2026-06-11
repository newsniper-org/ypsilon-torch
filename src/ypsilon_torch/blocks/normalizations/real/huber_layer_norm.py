from __future__ import annotations

import torch
from torch import Tensor, nn

from ypsilon_torch import FPDTypeIdx, get_float_dtype
from ypsilon_torch.functional import huber_location, median_abs_deviation


def _make_view_shape(ndim: int, dim: int) -> list[int]:
    """Build a broadcast shape with -1 at *dim* and 1 elsewhere."""
    shape = [1] * ndim
    shape[dim] = -1
    return shape


class HuberLayerNorm(nn.Module):
    r"""Layer normalization using a Huber M-estimator location and MAD scale.

    .. math::

        \mu_H = \operatorname*{arg\,min}_y \sum_j
                  \rho_\delta\!\left(\frac{x_j - y}{s}\right),
        \qquad s = \operatorname{MAD}_j(x)

        \operatorname{HuberLayerNorm}(x)_j
            = \gamma_j \frac{x_j - \mu_H}{s + \epsilon} + \beta_j

    The Huber location has bounded influence (quadratic for small residuals,
    linear beyond :math:`\delta`), giving a more outlier-robust centring than
    the mean while staying convex (unlike the redescending biweight). It sits
    between :class:`AsinhMeanLayerNorm` and :class:`RobustLayerNorm`.

    Parameters
    ----------
    normalized_shape : int
        Size of the dimension to normalize.
    delta : float
        Huber tuning constant in MAD-scaled units. Default ``1.345``
        (~95% Gaussian efficiency).
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
        delta: float = 1.345,
        eps: float = 1e-5,
        dim: int = -1,
        affine: bool = True,
        dtype_idx: FPDTypeIdx = 64,
        max_iter: int = 10,
    ) -> None:
        super().__init__()
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}")

        self.normalized_shape: int = normalized_shape
        self.delta: float = delta
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
        mu = huber_location(
            x, dim=self.dim, delta=self.delta, max_iter=self.max_iter, keepdim=True
        )
        s = median_abs_deviation(x, dim=self.dim, keepdim=True)

        x_norm = (x - mu) / (s + self.eps)

        if self.affine:
            view = _make_view_shape(x.ndim, self.dim)
            return self.gamma.view(*view) * x_norm + self.beta.view(*view)
        return x_norm
