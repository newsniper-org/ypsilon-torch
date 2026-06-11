"""Robust location and scale estimators (M-estimators and order-statistic based).

These complement :mod:`ypsilon_torch.functional.robust_stats` (Fréchet
median/medoid, arsinh-space statistics) with classical robust statistics:

- **Huber location** — a convex M-estimator that is quadratic for small
  residuals and linear for large ones (bounded influence).
- **Tukey biweight location** — a redescending M-estimator that fully
  rejects gross outliers (zero influence beyond the tuning constant).
- **Trimmed / Winsorized mean** — order-statistic estimators that drop or
  clamp a fraction of the most extreme samples.
- **Median absolute deviation (MAD)** and **biweight midvariance** — robust
  scale estimators.

All functions operate along a single ``dim`` and return a tensor with that
dimension reduced (``keepdim=False``), matching the convention of
:mod:`robust_stats`.  Real-valued inputs only.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ===================================================================
# Robust scale: MAD
# ===================================================================

# Consistency constant so that, for Gaussian data, MAD estimates sigma.
_MAD_GAUSSIAN_SCALE = 1.4826


def median_abs_deviation(
    x: Tensor,
    dim: int,
    scale: float = _MAD_GAUSSIAN_SCALE,
    keepdim: bool = False,
) -> Tensor:
    r"""Median absolute deviation: :math:`s \cdot \mathrm{med}_j |x_j - \mathrm{med}(x)|`.

    Parameters
    ----------
    x : Tensor
        Real-valued input.
    dim : int
        Dimension along which to reduce.
    scale : float
        Consistency constant.  Default ``1.4826`` makes the MAD a consistent
        estimator of the standard deviation for Gaussian data.  Use ``1.0``
        for the raw MAD.
    keepdim : bool
        Whether to keep the reduced dimension.
    """
    med = torch.median(x, dim=dim, keepdim=True).values
    mad = torch.median((x - med).abs(), dim=dim, keepdim=True).values * scale
    return mad if keepdim else mad.squeeze(dim)


# ===================================================================
# Huber location (convex M-estimator via IRLS)
# ===================================================================

def huber_location(
    x: Tensor,
    dim: int,
    delta: float = 1.345,
    max_iter: int = 20,
    tol: float = 1e-6,
    eps: float = 1e-12,
    keepdim: bool = False,
) -> Tensor:
    r"""Huber M-estimator of location, solved by IRLS.

    Minimises :math:`\sum_j \rho_\delta((x_j - \mu)/s)` where :math:`s` is a
    MAD scale and :math:`\rho_\delta` is the Huber loss (quadratic for
    :math:`|r| \le \delta`, linear beyond).  The IRLS weight is

    .. math::

        w_j = \min\!\left(1,\; \frac{\delta}{|r_j|}\right),
        \qquad r_j = (x_j - \mu)/s .

    Parameters
    ----------
    x : Tensor
        Real-valued input.
    dim : int
        Dimension along which to estimate the location.
    delta : float
        Huber tuning constant (in MAD-scaled residual units).  ``1.345``
        gives ~95% efficiency under Gaussian noise.
    max_iter, tol, eps
        IRLS iteration controls.
    keepdim : bool
        Whether to keep the reduced dimension.
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")

    mu = torch.median(x, dim=dim, keepdim=True).values
    s = median_abs_deviation(x, dim=dim, keepdim=True).clamp(min=eps)

    with torch.no_grad():
        for _ in range(max_iter):
            r = (x - mu) / s
            w = (delta / r.abs().clamp(min=eps)).clamp(max=1.0)
            mu_new = (w * x).sum(dim=dim, keepdim=True) / w.sum(dim=dim, keepdim=True)
            if (mu_new - mu).abs().max() < tol:
                mu = mu_new
                break
            mu = mu_new

    return mu if keepdim else mu.squeeze(dim)


# ===================================================================
# Tukey biweight location (redescending M-estimator via IRLS)
# ===================================================================

def tukey_biweight_location(
    x: Tensor,
    dim: int,
    c: float = 4.685,
    max_iter: int = 20,
    tol: float = 1e-6,
    eps: float = 1e-12,
    keepdim: bool = False,
) -> Tensor:
    r"""Tukey biweight (bisquare) M-estimator of location, solved by IRLS.

    Redescending estimator: residuals with :math:`|r| > c` receive **zero**
    weight, so gross outliers are fully rejected.  The IRLS weight is

    .. math::

        w_j = \left(1 - (r_j/c)^2\right)^2 \cdot \mathbb{1}[|r_j| \le c],
        \qquad r_j = (x_j - \mu)/s .

    Parameters
    ----------
    x : Tensor
        Real-valued input.
    dim : int
        Dimension along which to estimate the location.
    c : float
        Biweight tuning constant (in MAD-scaled residual units).  ``4.685``
        gives ~95% efficiency under Gaussian noise.
    max_iter, tol, eps
        IRLS iteration controls.
    keepdim : bool
        Whether to keep the reduced dimension.
    """
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")

    mu = torch.median(x, dim=dim, keepdim=True).values
    s = median_abs_deviation(x, dim=dim, keepdim=True).clamp(min=eps)

    with torch.no_grad():
        for _ in range(max_iter):
            r = (x - mu) / s
            u = (r / c).clamp(min=-1.0, max=1.0)
            w = (1.0 - u * u).pow(2)
            denom = w.sum(dim=dim, keepdim=True).clamp(min=eps)
            mu_new = (w * x).sum(dim=dim, keepdim=True) / denom
            if (mu_new - mu).abs().max() < tol:
                mu = mu_new
                break
            mu = mu_new

    return mu if keepdim else mu.squeeze(dim)


# ===================================================================
# Trimmed / Winsorized mean (order-statistic estimators)
# ===================================================================

def _resolve_trim_count(n: int, trim: float) -> int:
    if not (0.0 <= trim < 0.5):
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    return int(trim * n)


def trimmed_mean(x: Tensor, dim: int, trim: float = 0.1, keepdim: bool = False) -> Tensor:
    r"""Symmetric trimmed mean.

    Sorts along *dim*, discards the lowest and highest ``floor(trim * n)``
    samples, and averages the rest.

    Parameters
    ----------
    x : Tensor
        Real-valued input.
    dim : int
        Dimension along which to reduce.
    trim : float
        Fraction trimmed from *each* tail, in ``[0, 0.5)``.
    keepdim : bool
        Whether to keep the reduced dimension.
    """
    n = x.size(dim)
    k = _resolve_trim_count(n, trim)
    xs = torch.sort(x, dim=dim).values
    if k > 0:
        xs = xs.narrow(dim, k, n - 2 * k)
    out = xs.mean(dim=dim, keepdim=keepdim)
    return out


def winsorized_mean(x: Tensor, dim: int, trim: float = 0.1, keepdim: bool = False) -> Tensor:
    r"""Symmetric Winsorized mean.

    Sorts along *dim* and **clamps** (rather than discards) the lowest and
    highest ``floor(trim * n)`` samples to the nearest retained order
    statistic before averaging.

    Parameters
    ----------
    x : Tensor
        Real-valued input.
    dim : int
        Dimension along which to reduce.
    trim : float
        Fraction Winsorized at *each* tail, in ``[0, 0.5)``.
    keepdim : bool
        Whether to keep the reduced dimension.
    """
    n = x.size(dim)
    k = _resolve_trim_count(n, trim)
    xs = torch.sort(x, dim=dim).values
    if k > 0:
        lo = xs.narrow(dim, k, 1)
        hi = xs.narrow(dim, n - k - 1, 1)
        xs = torch.minimum(torch.maximum(xs, lo), hi)
    return xs.mean(dim=dim, keepdim=keepdim)


# ===================================================================
# Biweight midvariance (robust scale)
# ===================================================================

def biweight_midvariance(
    x: Tensor,
    dim: int,
    c: float = 9.0,
    eps: float = 1e-12,
    keepdim: bool = False,
) -> Tensor:
    r"""Tukey biweight midvariance — a robust estimator of variance.

    .. math::

        \zeta_{bi}^2 = \frac{n \sum_{|u_j|<1} (x_j-M)^2 (1-u_j^2)^4}
                            {\left(\sum_{|u_j|<1} (1-u_j^2)(1-5u_j^2)\right)^2},
        \qquad u_j = \frac{x_j - M}{c \cdot \mathrm{MAD}} .

    where :math:`M` is the median.  Returns the midvariance (square it is
    already; take ``sqrt`` for the biweight midstandard-deviation).

    Parameters
    ----------
    x : Tensor
        Real-valued input.
    dim : int
        Dimension along which to reduce.
    c : float
        Tuning constant (default ``9.0``).
    eps, keepdim
        Numerical floor and dimension-retention flag.
    """
    n = x.size(dim)
    med = torch.median(x, dim=dim, keepdim=True).values
    mad = torch.median((x - med).abs(), dim=dim, keepdim=True).values.clamp(min=eps)

    d = x - med
    u = d / (c * mad)
    mask = (u.abs() < 1.0).to(x.dtype)
    one_minus_u2 = (1.0 - u * u)

    num = n * (mask * d * d * one_minus_u2.pow(4)).sum(dim=dim, keepdim=True)
    den = (mask * one_minus_u2 * (1.0 - 5.0 * u * u)).sum(dim=dim, keepdim=True)
    out = num / (den * den).clamp(min=eps)
    return out if keepdim else out.squeeze(dim)
