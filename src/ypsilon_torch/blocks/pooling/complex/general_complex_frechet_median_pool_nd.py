from __future__ import annotations

from typing import Callable, Sequence

from torch import Tensor
from ypsilon_torch.functional import frechet_median

from ._complex_nd_pool_base import ComplexNDPoolBase


class ComplexFrechetMedianPoolND(ComplexNDPoolBase):
    r"""Complex N-dimensional Fréchet-median pooling with an arbitrary distance function.

    Parameters
    ----------
    dist_fn : ``Tensor → Tensor``
        Distance function :math:`\rho` applied to non-negative residual
        magnitudes.  Must be differentiable.
    kernel_size, stride, padding
        Passed to :class:`ComplexNDPoolBase`.
    max_iter : int
        Maximum Weiszfeld iterations. Default ``20``.
    tol : float
        Convergence tolerance. Default ``1e-6``.
    """

    def __init__(
        self,
        dist_fn: Callable[[Tensor], Tensor],
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> None:
        super().__init__(kernel_size, stride, padding)
        self.dist_fn = dist_fn
        self.max_iter: int = max_iter
        self.tol: float = tol

    def _reduce(self, windows: Tensor, dim: int) -> Tensor:
        return frechet_median(
            windows, dim=dim, dist_fn=self.dist_fn,
            max_iter=self.max_iter, tol=self.tol,
        )
