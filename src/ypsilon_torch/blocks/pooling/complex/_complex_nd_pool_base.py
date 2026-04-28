"""Base class for complex-valued N-dimensional pooling.

Reuses the window-extraction logic from :class:`NDPoolBase`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from torch import Tensor

from ..real._nd_pool_base import NDPoolBase


class ComplexNDPoolBase(NDPoolBase, ABC):
    """Abstract base for complex ND pooling layers.

    Inherits window extraction from :class:`NDPoolBase`.  The ``unfold``
    operation works identically on complex tensors.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
    ) -> None:
        super().__init__(kernel_size, stride, padding)

    @abstractmethod
    def _reduce(self, windows: Tensor, dim: int) -> Tensor:
        ...
