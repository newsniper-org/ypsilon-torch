"""Reusable auxiliary-loss base for sparse / routed blocks.

Several blocks in this library (Mixture-of-Depths, Mixture-of-Hidden-Dimensions,
Attractor Patch Networks, ...) emit an auxiliary regularization term during
``forward`` — a load-balancing or prediction loss that must be added to the
main objective.

:class:`AuxLossModule` provides exactly the ``reg_loss`` mechanism that
:class:`ypsilon_torch.blocks.bilinear_mlp.BilinearMLPBase` uses, factored into a
standalone base so routed blocks can share it. It is backed by
``torchutils.decorators.auxloss`` (the project's dependency), **not** a bespoke
reimplementation: the term lives in a non-persistent ``_reg_loss`` buffer, is
assigned via ``self.reg_loss = <tensor>`` during ``forward``, and is gathered
across a whole model with ``AuxLossModule.reg_loss.collect(model)``.

Because the collector walks every sub-module looking for a ``_reg_loss``
buffer, a single ``reg_loss.collect(model)`` call aggregates the contributions
of these blocks *and* of any :class:`BilinearMLPBase` in the same model.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import nn
from torchutils.decorators import auxloss


class AuxLossModule(nn.Module):
    """``nn.Module`` base exposing a collectable ``reg_loss`` (torchutils.auxloss).

    Subclasses call ``super().__init__()`` and then assign ``self.reg_loss``
    (a scalar tensor) during ``forward``; clear it with ``self.reg_loss = 0.0``.
    Collect across a model with ``AuxLossModule.reg_loss.collect(model)``.
    """

    def __init__(self) -> None:
        super().__init__()
        # Non-persistent: follows .to(device/dtype) but is never checkpointed.
        self.register_buffer("_reg_loss", torch.zeros(()), persistent=False)

    @auxloss
    def reg_loss(self) -> torch.Tensor:
        return self._reg_loss

    @reg_loss.setter
    def reg_loss(self, loss) -> None:
        # Accept a tensor contribution or a python scalar (e.g. ``= 0.0``).
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(
                float(loss), dtype=self._reg_loss.dtype, device=self._reg_loss.device
            )
        self._reg_loss = loss.to(dtype=self._reg_loss.dtype).reshape(())

    @reg_loss.collector
    def reg_loss(
        self,
        aggregate: Callable[[torch.Tensor], torch.Tensor] = torch.sum,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Aggregate ``_reg_loss`` over ``self`` and all sub-modules."""
        losses: List[torch.Tensor] = []
        for m in self.modules():
            rl = getattr(m, "_reg_loss", None)
            if isinstance(rl, torch.Tensor):
                losses.append(rl.reshape(()))
        if not losses:
            out = torch.zeros(
                (), dtype=self._reg_loss.dtype, device=self._reg_loss.device
            )
        else:
            out = aggregate(torch.stack(losses))
        if dtype is not None:
            out = out.to(dtype=dtype)
        if device is not None:
            out = out.to(device=device)
        return out.reshape(())

    @reg_loss.resetter
    @torch.no_grad()
    def reg_loss(self, value: float) -> None:
        """Reset ``_reg_loss`` on ``self`` and all sub-modules to ``value``."""
        for m in self.modules():
            rl = getattr(m, "_reg_loss", None)
            if isinstance(rl, torch.Tensor):
                setattr(
                    m, "_reg_loss", rl.detach().new_tensor(float(value)).reshape(())
                )
