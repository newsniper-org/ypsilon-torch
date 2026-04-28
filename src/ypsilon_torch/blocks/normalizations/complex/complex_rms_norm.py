from __future__ import annotations

import torch
from torch import Tensor, nn
from ypsilon_torch import FPDTypeIdx, get_float_dtype


def _make_view_shape(ndim: int, dim: int) -> list[int]:
    """Build a broadcast shape with -1 at *dim* and 1 elsewhere."""
    shape = [1] * ndim
    shape[dim] = -1
    return shape


class ComplexRMSNorm(nn.Module):
    """RMSNorm for complex-valued tensors.

    Normalizes by the RMS of the magnitude, preserving phase.
    Applies a real-valued learnable weight per element along *dim*.

    Parameters
    ----------
    normalized_shape : int
        Size of the dimension to normalize.
    eps : float
        Stability epsilon. Default ``1e-6``.
    dim : int
        Dimension to normalize over. Default ``-1``.
    affine : bool
        If True, apply a learnable weight. Default False.
    dtype_idx : FPDTypeIdx
        Floating-point precision index. Default ``64``.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        dim: int = -1,
        affine: bool = False,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        self.normalized_shape: int = normalized_shape
        self.eps: float = eps
        self.dim: int = dim
        self.affine: bool = affine

        if affine:
            self.weight: nn.Parameter = nn.Parameter(
                torch.empty(normalized_shape, dtype=get_float_dtype(dtype_idx))
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)

    def forward(self, z: Tensor) -> Tensor:
        """Normalize and optionally scale.

        Parameters
        ----------
        z : Tensor
            Complex tensor. The size along ``self.dim`` must equal
            ``self.normalized_shape``.

        Returns
        -------
        Tensor
            Same shape, complex.
        """
        rms = torch.sqrt(
            torch.mean(z.real.pow(2) + z.imag.pow(2), dim=self.dim, keepdim=True)
            + self.eps
        )
        if self.affine:
            w = self.weight.view(*_make_view_shape(z.ndim, self.dim))
            return z * (w / rms)
        return z / rms
