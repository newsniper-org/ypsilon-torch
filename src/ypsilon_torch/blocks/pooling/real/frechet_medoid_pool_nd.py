from __future__ import annotations

from typing import Sequence

from torch import Tensor
from ypsilon_torch.functional import frechet_medoid_lp

from ._nd_pool_base import NDPoolBase


class FrechetMedoidLpPoolND(NDPoolBase):
    r"""N-dimensional Fréchet-medoid pooling with L1 distance.

    Selects the window element that minimises the sum of absolute distances
    to all others.  The result is detached from the computation graph.

    Parameters
    ----------
    kernel_size, stride, padding
        Passed to :class:`NDPoolBase`.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
    ) -> None:
        super().__init__(kernel_size, stride, padding)

    def _reduce(self, windows: Tensor, dim: int) -> Tensor:
        return frechet_medoid_lp(windows, dim=dim)
