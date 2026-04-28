from __future__ import annotations

from typing import Sequence

from torch import Tensor
from ypsilon_torch.functional import root_frechet_medoid_lp_square

from ._complex_nd_pool_base import ComplexNDPoolBase


class ComplexRootFrechetMedoidLpSquarePoolND(ComplexNDPoolBase):
    r"""Complex :math:`\sqrt{\mathrm{FréchetMedoid}_{L_1}(Z^2)}` pooling.

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
        return root_frechet_medoid_lp_square(windows, dim=dim)
