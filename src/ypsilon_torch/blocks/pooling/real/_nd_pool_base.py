"""Base class for N-dimensional pooling via window extraction + reduction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import Tensor, nn


def _to_tuple(value: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * ndim
    return tuple(value)


class NDPoolBase(nn.Module, ABC):
    """Abstract base for ND pooling layers.

    Subclasses only need to implement :meth:`_reduce` which collapses the
    window dimension ``K`` (last axis) into a single value.

    Parameters
    ----------
    kernel_size : int or tuple[int, ...]
        Size of the pooling window.
    stride : int, tuple[int, ...], or None
        Stride of the pooling window.  ``None`` means same as *kernel_size*.
    padding : int or tuple[int, ...]
        Zero-padding added to each spatial dimension.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
    ) -> None:
        super().__init__()
        # Store raw values; will be expanded to tuples in forward() once we
        # know the number of spatial dimensions.
        self._kernel_size_raw = kernel_size
        self._stride_raw = stride
        self._padding_raw = padding

    # ------------------------------------------------------------------
    def _extract_windows(self, x: Tensor) -> Tensor:
        """Extract local windows from *x*.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, C, *spatial)``.

        Returns
        -------
        Tensor
            Shape ``(B, C, *out_spatial, K)`` where
            ``K = prod(kernel_size)``.
        """
        n_spatial = x.ndim - 2
        ks = _to_tuple(self._kernel_size_raw, n_spatial)
        stride = _to_tuple(
            self._stride_raw if self._stride_raw is not None else self._kernel_size_raw,
            n_spatial,
        )
        pad = _to_tuple(self._padding_raw, n_spatial)

        # Apply zero-padding if needed
        if any(p > 0 for p in pad):
            pad_args: list[int] = []
            for p in reversed(pad):
                pad_args.extend([p, p])
            x = torch.nn.functional.pad(x, pad_args)

        # Chain .unfold() for each spatial dim
        for i, (k, s) in enumerate(zip(ks, stride)):
            x = x.unfold(2 + i, k, s)      # new dim appended at the end

        # After unfolding d spatial dims the shape is
        #   (B, C, O1, O2, ..., Od, K1, K2, ..., Kd)
        # We want to flatten the K dims into one.
        n_k_dims = n_spatial
        if n_k_dims > 1:
            x = x.flatten(start_dim=-n_k_dims)  # → (..., K)

        return x

    # ------------------------------------------------------------------
    @abstractmethod
    def _reduce(self, windows: Tensor, dim: int) -> Tensor:
        """Reduce the window dimension *dim* to a single value."""
        ...

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        windows = self._extract_windows(x)          # (B, C, *out, K)
        return self._reduce(windows, dim=-1)         # (B, C, *out)
