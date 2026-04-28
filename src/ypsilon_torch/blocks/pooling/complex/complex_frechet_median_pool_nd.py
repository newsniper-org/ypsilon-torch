from __future__ import annotations

from typing import Sequence

from torch import Tensor
from ypsilon_torch.functional import frechet_median_lp

from ._complex_nd_pool_base import ComplexNDPoolBase


class ComplexFrechetMedianLpPoolND(ComplexNDPoolBase):
    r"""Complex N-dimensional Fréchet-median pooling with modulus distance (Weiszfeld in C).

    Parameters
    ----------
    kernel_size, stride, padding
        Passed to :class:`ComplexNDPoolBase`.
    max_iter : int
        Maximum Weiszfeld iterations. Default ``20``.
    tol : float
        Convergence tolerance. Default ``1e-6``.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> None:
        super().__init__(kernel_size, stride, padding)
        self.max_iter: int = max_iter
        self.tol: float = tol

    def _reduce(self, windows: Tensor, dim: int) -> Tensor:
        return frechet_median_lp(
            windows, dim=dim, max_iter=self.max_iter, tol=self.tol
        )
