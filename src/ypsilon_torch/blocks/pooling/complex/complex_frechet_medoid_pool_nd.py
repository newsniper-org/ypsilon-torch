from __future__ import annotations

from typing import Sequence

from torch import Tensor
from ypsilon_torch.functional import frechet_medoid_lp

from ._complex_nd_pool_base import ComplexNDPoolBase


class ComplexFrechetMedoidLpPoolND(ComplexNDPoolBase):
    r"""Complex N-dimensional Fréchet-medoid pooling with modulus distance.

    Uses complex modulus as the distance function.
    The result is detached from the computation graph.

    Parameters
    ----------
    kernel_size, stride, padding
        Passed to :class:`ComplexNDPoolBase`.
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
