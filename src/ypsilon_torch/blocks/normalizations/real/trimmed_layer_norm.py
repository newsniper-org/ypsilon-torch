from __future__ import annotations

import torch
from torch import Tensor, nn

from ypsilon_torch import FPDTypeIdx, get_float_dtype
from ypsilon_torch.functional import trimmed_mean, median_abs_deviation


def _make_view_shape(ndim: int, dim: int) -> list[int]:
    """Build a broadcast shape with -1 at *dim* and 1 elsewhere."""
    shape = [1] * ndim
    shape[dim] = -1
    return shape


class TrimmedLayerNorm(nn.Module):
    r"""Layer normalization using a trimmed-mean location and MAD scale.

    .. math::

        \mu_T = \operatorname{TrimmedMean}_{\tau}(x),
        \qquad s = \operatorname{MAD}_j(x)

        \operatorname{TrimmedLayerNorm}(x)_j
            = \gamma_j \frac{x_j - \mu_T}{s + \epsilon} + \beta_j

    The location discards the lowest and highest ``floor(tau * n)`` samples
    before averaging, giving a simple breakdown-bounded centring. The scale
    uses the median absolute deviation. Unlike the IRLS-based
    :class:`RobustLayerNorm` / :class:`HuberLayerNorm`, the location here is a
    closed-form order statistic (a single sort), so it is cheap and fully
    differentiable through the retained samples.

    Parameters
    ----------
    normalized_shape : int
        Size of the dimension to normalize.
    trim : float
        Fraction trimmed from each tail, in ``[0, 0.5)``. Default ``0.1``.
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
        trim: float = 0.1,
        eps: float = 1e-5,
        dim: int = -1,
        affine: bool = True,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        if not (0.0 <= trim < 0.5):
            raise ValueError(f"trim must be in [0, 0.5), got {trim}")

        self.normalized_shape: int = normalized_shape
        self.trim: float = trim
        self.eps: float = eps
        self.dim: int = dim
        self.affine: bool = affine

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
        mu = trimmed_mean(x, dim=self.dim, trim=self.trim, keepdim=True)
        s = median_abs_deviation(x, dim=self.dim, keepdim=True)

        x_norm = (x - mu) / (s + self.eps)

        if self.affine:
            view = _make_view_shape(x.ndim, self.dim)
            return self.gamma.view(*view) * x_norm + self.beta.view(*view)
        return x_norm
