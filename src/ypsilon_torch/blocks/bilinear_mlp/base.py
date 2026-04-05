"""Abstract base class for bilinear-MLP blocks.

A bilinear MLP computes ``y = W_out ((W_1 x) * (W_2 x))`` with no element-wise
nonlinearity between the two linear branches. The whole layer is therefore a
3rd-order tensor ``B_{ijk}`` in ``x``, which makes weight-based
interpretability possible (Pearce et al., 2024).

:class:`BilinearMLPBase` formalises this structure: subclasses implement the
two linear branches (``branch_a``, ``branch_b``) while the common
``forward`` / output projection / dropout / auxiliary-loss plumbing is shared.
"""

from __future__ import annotations

import abc
from typing import Callable, List, Optional

import torch
from torch import nn
from torchutils.decorators import auxloss


class BilinearMLPBase(nn.Module, abc.ABC):
    """Abstract base for bilinear-MLP blocks.

    Subclasses must implement :meth:`branch_a` and :meth:`branch_b`, each
    mapping an input of shape ``(B, N, dim)`` to ``(B, N, hidden_dim)``.

    Parameters
    ----------
    dim:
        Input feature dimension ``C_in``.
    hidden_dim:
        Bilinear-product dimension ``C_hidden``. If ``None``, defaults to
        ``int(dim * expansion)``.
    out_dim:
        Output feature dimension ``C_out``. Defaults to ``dim``.
    expansion:
        Multiplier used to compute ``hidden_dim`` when it is not given.
    dropout:
        Dropout probability applied to the bilinear product before the
        output projection.
    bias:
        Whether the output projection uses a bias term.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        expansion: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(round(dim * expansion))
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.expansion = expansion
        self.dropout_p = dropout

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, out_dim, bias=bias)

        # Auxiliary-loss slot. Registered as a non-persistent buffer so it
        # follows module .to(device)/.to(dtype) transitions automatically,
        # but is never saved to a checkpoint.
        self.register_buffer(
            "_reg_loss", torch.zeros(()), persistent=False,
        )

    # ------------------------------------------------------------------
    # Abstract branches: subclasses must override.
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def branch_a(self, x: torch.Tensor) -> torch.Tensor:
        """First linear branch; ``(B, N, dim) -> (B, N, hidden_dim)``."""

    @abc.abstractmethod
    def branch_b(self, x: torch.Tensor) -> torch.Tensor:
        """Second linear branch; ``(B, N, dim) -> (B, N, hidden_dim)``."""

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``W_out (branch_a(x) * branch_b(x))``.

        Subclasses may override :meth:`_compute_aux_loss` to contribute a
        regularization term to ``self.reg_loss``; by default no aux loss is
        computed and ``self.reg_loss`` is reset to ``0``.
        """
        a = self.branch_a(x)
        b = self.branch_b(x)
        self._compute_aux_loss(a, b)
        h = a * b
        h = self.dropout(h)
        return self.out_proj(h)

    def _compute_aux_loss(
        self, a: torch.Tensor, b: torch.Tensor,
    ) -> None:
        """Hook for subclasses to set ``self.reg_loss`` during forward.

        The default implementation resets the stored loss to zero on the
        current device/dtype so that :meth:`reg_loss.collect` remains
        well-defined regardless of training state.
        """
        self.reg_loss = torch.zeros(
            (), dtype=self._reg_loss.dtype, device=self._reg_loss.device,
        )

    # ------------------------------------------------------------------
    # Auxiliary-loss property (torchutils.auxloss, >= v0.1.1 required).
    # ------------------------------------------------------------------
    @auxloss
    def reg_loss(self) -> torch.Tensor:
        return self._reg_loss

    @reg_loss.setter
    def reg_loss(self, loss) -> None:
        # Accept either a torch.Tensor contribution or a Python scalar
        # (commonly ``model.reg_loss = 0.0`` to clear the slot, matching
        # the framesmoothie convention).
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(
                float(loss),
                dtype=self._reg_loss.dtype,
                device=self._reg_loss.device,
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
                (), dtype=self._reg_loss.dtype,
                device=self._reg_loss.device,
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
                    m, "_reg_loss",
                    rl.detach().new_tensor(float(value)).reshape(()),
                )
