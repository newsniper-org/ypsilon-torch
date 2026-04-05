"""AFBO-style asymmetric spatial-channel factorized bilinear MLP block.

Combines:

- the **Bilinear MLP** structure of Pearce et al. (2024) — a gated-linear
  unit without element-wise nonlinearity, so the layer is a 3rd-order tensor
  in the input (weight-analyzable);
- the **SCFBO** spatial / channel factorization of AFBO (ICLR 2025) — each
  branch is ``CM(SM(x))``, where ``SM`` is a per-channel spatial operator
  and ``CM`` is a structured-sparse channel mapping;
- the **asymmetric** channel-mapping design of AFBO — branch A uses
  :class:`GroupedCrossChannelMapping` (GCCM) and branch B uses
  :class:`OverlappedCycleChannelMapping` (OCCM), giving complementary
  cross-group / local-cyclic connectivity patterns.

The resulting module is a drop-in replacement for a ViT FFN.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import BilinearMLPBase
from .channel_mapping import (
    GroupedCrossChannelMapping,
    OverlappedCycleChannelMapping,
)
from .spatial_modeling import SMKind, SpatialModeling


class AsymmetricSpatialChannelFactorizedBilinearMLP(BilinearMLPBase):
    """Asymmetric SCFBO bilinear MLP block.

    Parameters
    ----------
    dim, hidden_dim, out_dim, expansion, dropout, bias:
        Forwarded to :class:`BilinearMLPBase`. ``hidden_dim`` and
        ``out_dim`` must both be divisible by ``num_groups``.
    sm_kind:
        Spatial-modeling operator used by both branches (``"dwconv"``,
        ``"pool"`` or ``"none"``).
    sm_kernel_size:
        Kernel size for the spatial modeling operator.
    num_groups:
        Number of channel groups ``G`` used by GCCM and OCCM.
    occm_overlap:
        Cyclic overlap ``s`` for OCCM.
    aux_loss_weight:
        Weight of the branch-decorrelation regularization term written to
        :attr:`reg_loss` during training. ``0.0`` disables the contribution
        while leaving the ``reg_loss`` property fully functional, so callers
        can always collect it via ``BilinearMLPBase.reg_loss.collect(model)``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        expansion: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
        sm_kind: SMKind = "dwconv",
        sm_kernel_size: int = 3,
        num_groups: int = 8,
        occm_overlap: int = 1,
        aux_loss_weight: float = 0.0,
    ) -> None:
        super().__init__(
            dim=dim, hidden_dim=hidden_dim, out_dim=out_dim,
            expansion=expansion, dropout=dropout, bias=bias,
        )
        self.num_groups = num_groups
        self.aux_loss_weight = float(aux_loss_weight)

        self.sm_a = SpatialModeling(
            dim=dim, kind=sm_kind, kernel_size=sm_kernel_size,
        )
        self.sm_b = SpatialModeling(
            dim=dim, kind=sm_kind, kernel_size=sm_kernel_size,
        )
        self.cm_a = GroupedCrossChannelMapping(
            in_channels=dim, out_channels=self.hidden_dim,
            num_groups=num_groups, bias=bias,
        )
        self.cm_b = OverlappedCycleChannelMapping(
            in_channels=dim, out_channels=self.hidden_dim,
            num_groups=num_groups, overlap=occm_overlap, bias=bias,
        )

    def branch_a(self, x: torch.Tensor) -> torch.Tensor:
        return self.cm_a(self.sm_a(x))

    def branch_b(self, x: torch.Tensor) -> torch.Tensor:
        return self.cm_b(self.sm_b(x))

    def _compute_aux_loss(
        self, a: torch.Tensor, b: torch.Tensor,
    ) -> None:
        """Penalize collinearity between the two bilinear branches.

        A squared per-token cosine similarity encourages ``a`` and ``b`` to
        capture complementary directions; otherwise the product ``a * b``
        collapses toward a squared-linear layer.
        """
        if self.aux_loss_weight == 0.0 or not self.training:
            # Reset to a 0 scalar on the correct device/dtype.
            self.reg_loss = torch.zeros(
                (), dtype=self._reg_loss.dtype,
                device=self._reg_loss.device,
            )
            return
        eps = 1e-8
        # Cosine along the channel axis, per (batch, token).
        dot = (a * b).sum(dim=-1)
        na = a.norm(dim=-1)
        nb = b.norm(dim=-1)
        cos = dot / (na * nb + eps)
        loss = self.aux_loss_weight * cos.pow(2).mean()
        self.reg_loss = loss
