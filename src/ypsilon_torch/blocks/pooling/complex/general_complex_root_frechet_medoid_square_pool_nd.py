from __future__ import annotations

from typing import Callable, Sequence

from torch import Tensor
from ypsilon_torch.functional import root_frechet_medoid_square

from ._complex_nd_pool_base import ComplexNDPoolBase


class ComplexRootFrechetMedoidSquarePoolND(ComplexNDPoolBase):
    r"""Complex :math:`\sqrt{\mathrm{FréchetMedoid}_\rho(Z^2)}` pooling
    with an arbitrary distance function.

    The result is detached from the computation graph.

    Parameters
    ----------
    dist_fn : ``Tensor → Tensor``
        Distance function :math:`\rho` applied to non-negative residual
        magnitudes.
    kernel_size, stride, padding
        Passed to :class:`ComplexNDPoolBase`.
    """

    def __init__(
        self,
        dist_fn: Callable[[Tensor], Tensor],
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
    ) -> None:
        super().__init__(kernel_size, stride, padding)
        self.dist_fn = dist_fn

    def _reduce(self, windows: Tensor, dim: int) -> Tensor:
        return root_frechet_medoid_square(windows, dim=dim, dist_fn=self.dist_fn)
