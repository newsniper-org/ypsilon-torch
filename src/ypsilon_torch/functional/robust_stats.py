"""Robust statistics primitives: Fréchet median/medoid and arsinh-space statistics."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


# ===================================================================
# Internal IRLS / Weiszfeld solvers
# ===================================================================

def _irls_median_lp(
    x: Tensor,
    dim: int,
    p: float,
    max_iter: int,
    tol: float,
    eps: float = 1e-12,
) -> Tensor:
    """IRLS solver for the L_p Fréchet median on the real line (1 < p < 2)."""
    mu = torch.median(x, dim=dim).values.unsqueeze(dim)

    with torch.no_grad():
        for _ in range(max_iter):
            residual = (x - mu).abs().clamp(min=eps)
            w = residual.pow(p - 2)
            mu_new = (w * x).sum(dim=dim, keepdim=True) / w.sum(dim=dim, keepdim=True)
            if (mu_new - mu).abs().max() < tol:
                break
            mu = mu_new

    return mu.squeeze(dim)


def _weiszfeld_complex_lp(
    z: Tensor,
    dim: int,
    max_iter: int,
    tol: float,
    eps: float = 1e-12,
) -> Tensor:
    """Weiszfeld algorithm for the geometric median in the complex plane."""
    mu = z.mean(dim=dim).unsqueeze(dim)

    with torch.no_grad():
        for _ in range(max_iter):
            residual = (z - mu).abs().clamp(min=eps)
            w = 1.0 / residual
            mu_new = (w * z).sum(dim=dim, keepdim=True) / w.sum(dim=dim, keepdim=True)
            if (mu_new - mu).abs().max() < tol:
                break
            mu = mu_new

    return mu.squeeze(dim)


def _irls_general(
    x: Tensor,
    dim: int,
    dist_fn: Callable[[Tensor], Tensor],
    max_iter: int,
    tol: float,
    eps: float = 1e-12,
) -> Tensor:
    """IRLS solver with an arbitrary distance function rho(|r|).

    Weights are derived via autodiff: w_j = rho'(r_j) / r_j.
    """
    mu = torch.median(x, dim=dim).values.unsqueeze(dim)

    with torch.no_grad():
        for _ in range(max_iter):
            r = (x - mu).abs().clamp(min=eps)
            # Enable grad locally for autodiff on dist_fn
            with torch.enable_grad():
                r_grad = r.detach().requires_grad_(True)
                rho = dist_fn(r_grad)
                (d_rho,) = torch.autograd.grad(rho.sum(), r_grad)
            w = d_rho / r
            mu_new = (w * x).sum(dim=dim, keepdim=True) / w.sum(dim=dim, keepdim=True)
            if (mu_new - mu).abs().max() < tol:
                break
            mu = mu_new

    return mu.squeeze(dim)


def _weiszfeld_general_complex(
    z: Tensor,
    dim: int,
    dist_fn: Callable[[Tensor], Tensor],
    max_iter: int,
    tol: float,
    eps: float = 1e-12,
) -> Tensor:
    """Weiszfeld algorithm with an arbitrary distance function in C.

    dist_fn operates on real-valued magnitudes |z_j - mu|.
    """
    mu = z.mean(dim=dim).unsqueeze(dim)

    with torch.no_grad():
        for _ in range(max_iter):
            r = (z - mu).abs().clamp(min=eps)
            with torch.enable_grad():
                r_grad = r.detach().requires_grad_(True)
                rho = dist_fn(r_grad)
                (d_rho,) = torch.autograd.grad(rho.sum(), r_grad)
            w = d_rho / r
            mu_new = (w * z).sum(dim=dim, keepdim=True) / w.sum(dim=dim, keepdim=True)
            if (mu_new - mu).abs().max() < tol:
                break
            mu = mu_new

    return mu.squeeze(dim)


# ===================================================================
# Fréchet median — L_p specialisation
# ===================================================================

def frechet_median_lp(
    x: Tensor,
    dim: int,
    p: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Tensor:
    r"""Fréchet median with L_p distance: :math:`\arg\min_y \sum_j |x_j - y|^p`.

    Parameters
    ----------
    x : Tensor
        Input tensor (real or complex).
    dim : int
        Dimension along which to compute the median.
    p : float
        Distance exponent.
        * ``p = 1``: ordinary median.
        * ``p = 2``: arithmetic mean.
        * ``1 < p < 2``: IRLS solver.
        Complex tensors always use modulus distance (p is ignored).
    max_iter : int
        Maximum IRLS / Weiszfeld iterations.
    tol : float
        Convergence tolerance.
    """
    if x.is_complex():
        return _weiszfeld_complex_lp(z=x, dim=dim, max_iter=max_iter, tol=tol)

    if p == 1.0:
        return torch.median(x, dim=dim).values
    elif p == 2.0:
        return torch.mean(x, dim=dim)

    if not (1.0 <= p <= 2.0):
        raise ValueError(f"p must be 1, 2, or in (1, 2), got {p}")

    return _irls_median_lp(x=x, dim=dim, p=p, max_iter=max_iter, tol=tol)


# ===================================================================
# Fréchet median — general distance
# ===================================================================

def frechet_median(
    x: Tensor,
    dim: int,
    dist_fn: Callable[[Tensor], Tensor],
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Tensor:
    r"""Fréchet median with an arbitrary distance function.

    .. math::

        \arg\min_y \sum_j \rho(|x_j - y|)

    IRLS weights are computed via autodiff on *dist_fn*.

    Parameters
    ----------
    x : Tensor
        Input tensor (real or complex).
    dim : int
        Dimension along which to compute the median.
    dist_fn : ``Tensor → Tensor``
        Distance function :math:`\rho` applied to non-negative residual
        magnitudes.  Must be differentiable.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance.
    """
    if x.is_complex():
        return _weiszfeld_general_complex(
            z=x, dim=dim, dist_fn=dist_fn, max_iter=max_iter, tol=tol,
        )
    return _irls_general(
        x=x, dim=dim, dist_fn=dist_fn, max_iter=max_iter, tol=tol,
    )


# ===================================================================
# Fréchet medoid — L_p specialisation
# ===================================================================

def _medoid_core(x_moved: Tensor, dist_fn: Callable[[Tensor], Tensor] | None) -> Tensor:
    """Shared medoid logic.  *x_moved* has the target dim moved to -1."""
    xi = x_moved.unsqueeze(-1)
    xj = x_moved.unsqueeze(-2)
    pairwise = (xi - xj).abs()                      # real magnitudes
    if dist_fn is not None:
        pairwise = dist_fn(pairwise)
    total_dist = pairwise.sum(dim=-1)                # (..., N)

    idx = total_dist.argmin(dim=-1, keepdim=True)
    result = x_moved.gather(-1, idx).squeeze(-1)
    return result.detach()


def frechet_medoid_lp(x: Tensor, dim: int) -> Tensor:
    r"""Fréchet medoid with L1 distance (default).

    Picks the sample element that minimises the sum of absolute distances
    to all others.  The result is **detached** from the computation graph.

    Parameters
    ----------
    x : Tensor
        Input tensor (real or complex).
    dim : int
        Dimension along which to compute the medoid.
    """
    return _medoid_core(x.movedim(dim, -1), dist_fn=None)


# ===================================================================
# Fréchet medoid — general distance
# ===================================================================

def frechet_medoid(
    x: Tensor,
    dim: int,
    dist_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    r"""Fréchet medoid with an arbitrary distance function.

    .. math::

        \arg\min_{y \in X} \sum_j \rho(|x_j - y|)

    The result is **detached** from the computation graph.

    Parameters
    ----------
    x : Tensor
        Input tensor (real or complex).
    dim : int
        Dimension along which to compute the medoid.
    dist_fn : ``Tensor → Tensor``
        Distance function :math:`\rho` applied to non-negative residual
        magnitudes.
    """
    return _medoid_core(x.movedim(dim, -1), dist_fn=dist_fn)


# ===================================================================
# arsinh-space statistics  (real-valued only)
# ===================================================================

def sinh_mean_arsinh(x: Tensor, c: Tensor | float, dim: int) -> Tensor:
    r"""SinhMeanArsinh: :math:`c \sinh(\mathbb{E}[\operatorname{arsinh}(x/c)])`."""
    z = torch.asinh(x / c)
    return c * torch.sinh(z.mean(dim=dim))


def sinh_rms_arsinh(x: Tensor, c: Tensor | float, dim: int) -> Tensor:
    r"""SinhRMSArsinh: :math:`c \sinh(\operatorname{RMS}[\operatorname{arsinh}(x/c)])`."""
    z = torch.asinh(x / c)
    rms = torch.sqrt(z.pow(2).mean(dim=dim))
    return c * torch.sinh(rms)


def sinh_frechet_median_lp_arsinh(
    x: Tensor,
    c: Tensor | float,
    dim: int,
    p: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Tensor:
    r"""SinhFréchetMedianArsinh (L_p): arsinh → Fréchet median (L_p) → sinh."""
    z = torch.asinh(x / c)
    med = frechet_median_lp(z, dim=dim, p=p, max_iter=max_iter, tol=tol)
    return c * torch.sinh(med)


def sinh_frechet_median_arsinh(
    x: Tensor,
    c: Tensor | float,
    dim: int,
    dist_fn: Callable[[Tensor], Tensor],
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Tensor:
    r"""SinhFréchetMedianArsinh (general): arsinh → Fréchet median → sinh."""
    z = torch.asinh(x / c)
    med = frechet_median(z, dim=dim, dist_fn=dist_fn, max_iter=max_iter, tol=tol)
    return c * torch.sinh(med)


def sinh_frechet_medoid_lp_arsinh(
    x: Tensor,
    c: Tensor | float,
    dim: int,
) -> Tensor:
    r"""SinhFréchetMedoidArsinh (L1): arsinh → Fréchet medoid → sinh."""
    z = torch.asinh(x / c)
    med = frechet_medoid_lp(z, dim=dim)
    return c * torch.sinh(med)


def sinh_frechet_medoid_arsinh(
    x: Tensor,
    c: Tensor | float,
    dim: int,
    dist_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    r"""SinhFréchetMedoidArsinh (general): arsinh → Fréchet medoid → sinh."""
    z = torch.asinh(x / c)
    med = frechet_medoid(z, dim=dim, dist_fn=dist_fn)
    return c * torch.sinh(med)


# ===================================================================
# Root-Fréchet-*-Square
# ===================================================================

def root_frechet_median_lp_square(
    x: Tensor,
    dim: int,
    p: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Tensor:
    r""":math:`\sqrt{\operatorname{FréchetMedian}_{L_p}(X^2)}`."""
    return torch.sqrt(frechet_median_lp(x.pow(2), dim=dim, p=p, max_iter=max_iter, tol=tol))


def root_frechet_median_square(
    x: Tensor,
    dim: int,
    dist_fn: Callable[[Tensor], Tensor],
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Tensor:
    r""":math:`\sqrt{\operatorname{FréchetMedian}_{\rho}(X^2)}`."""
    return torch.sqrt(frechet_median(x.pow(2), dim=dim, dist_fn=dist_fn, max_iter=max_iter, tol=tol))


def root_frechet_medoid_lp_square(x: Tensor, dim: int) -> Tensor:
    r""":math:`\sqrt{\operatorname{FréchetMedoid}_{L_1}(X^2)}`."""
    return torch.sqrt(frechet_medoid_lp(x.pow(2), dim=dim))


def root_frechet_medoid_square(
    x: Tensor,
    dim: int,
    dist_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    r""":math:`\sqrt{\operatorname{FréchetMedoid}_{\rho}(X^2)}`."""
    return torch.sqrt(frechet_medoid(x.pow(2), dim=dim, dist_fn=dist_fn))
