from __future__ import annotations

from typing import Sequence

from torch import Tensor
from ypsilon_torch.functional import root_frechet_median_lp_square

from ._nd_pool_base import NDPoolBase


class RootFrechetMedianLpSquarePoolND(NDPoolBase):
    r"""N-dimensional :math:`\sqrt{\mathrm{FréchetMedian}_{L_p}(X^2)}` pooling.

    Parameters
    ----------
    kernel_size, stride, padding
        Passed to :class:`NDPoolBase`.
    p : float
        Distance exponent for the inner Fréchet median. Default ``1.0``.
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
        return root_frechet_median_lp_square(
            windows, dim=dim, p=self.p, max_iter=self.max_iter, tol=self.tol
        )
