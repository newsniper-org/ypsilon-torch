from __future__ import annotations

from typing import Sequence

from torch import Tensor
from ypsilon_torch.functional import frechet_median_lp

from ._nd_pool_base import NDPoolBase


class FrechetMedianLpPoolND(NDPoolBase):
    r"""N-dimensional Fréchet-median pooling with L_p distance.

    Parameters
    ----------
    kernel_size, stride, padding
        Passed to :class:`NDPoolBase`.
    p : float
        Distance exponent (``1`` for L1 median, ``(1,2)`` for IRLS).
    max_iter : int
        Maximum IRLS iterations. Default ``20``.
    tol : float
        Convergence tolerance. Default ``1e-6``.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
        p: float = 1.0,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> None:
        super().__init__(kernel_size, stride, padding)
        self.p: float = p
        self.max_iter: int = max_iter
        self.tol: float = tol

    def _reduce(self, windows: Tensor, dim: int) -> Tensor:
        return frechet_median_lp(
            windows, dim=dim, p=self.p, max_iter=self.max_iter, tol=self.tol
        )
