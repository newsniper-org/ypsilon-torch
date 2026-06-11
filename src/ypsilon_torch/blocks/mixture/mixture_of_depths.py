"""Mixture-of-Depths (MoD) — token-level dynamic compute allocation.

Reference
---------
Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro,
*Mixture-of-Depths: Dynamically allocating compute in transformer-based
language models* (arXiv:2404.02258).

A per-layer router scores every token; only the top-``C`` tokens (capacity
``C < N``) are processed by the wrapped block, while the rest bypass it via the
residual path — so each token costs a *different* amount of compute, but the
total per layer is fixed and known ahead of time.

Routing is **expert-choice** (the block picks its top-``C`` tokens), which
load-balances automatically and needs no balancing loss. Because top-``k`` over
the sequence is non-causal, an auxiliary **stop-gradient predictor** is trained
(binary cross-entropy against top-``k`` membership) so the routing decision can
be reproduced per-token at autoregressive inference time; its loss is exposed
via the ``reg_loss`` property (:class:`AuxLossModule`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._aux_loss import AuxLossModule


class MixtureOfDepths(AuxLossModule):
    r"""Wrap a block so only the router's top-``C`` tokens pass through it.

    .. math::

        x_i^{l+1} = \begin{cases}
            r_i\, f(\tilde X)_i + x_i & r_i > P_\beta(R) \\
            x_i & \text{otherwise}
        \end{cases}
        \qquad \beta = 1 - C/N

    where ``f`` is the wrapped block, ``r_i = w_\theta^\top x_i`` the router
    weight, and ``\tilde X`` the selected top-``C`` tokens.

    Parameters
    ----------
    block : nn.Module
        The wrapped block, mapping ``(B, C, d) -> (B, C, d)`` (e.g. a
        transformer layer or an FFN). It only ever sees the selected tokens.
    dim : int
        Token dimension ``d``.
    capacity_ratio : float
        Fraction ``C/N`` of tokens processed. Default ``0.125`` (the paper's
        recommended 12.5%).
    aux_weight : float
        Weight of the predictor BCE loss written to :attr:`aux_loss`.
        Default ``1.0``. Set ``0.0`` to disable the contribution (the
        predictor is still trained-compatible and usable at inference).
    predictor_hidden : int, optional
        Hidden width of the causal routing predictor MLP. Defaults to ``dim``.

    Notes
    -----
    ``forward`` uses expert-choice top-``k`` over the full sequence (the regime
    in which the paper trains and evaluates), and trains the predictor as a
    side task. For autoregressive *generation*, route each token with
    :meth:`predict_route` (``predictor(x) > 0``), which needs no future tokens.
    """

    def __init__(
        self,
        block: nn.Module,
        dim: int,
        capacity_ratio: float = 0.125,
        aux_weight: float = 1.0,
        predictor_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 < capacity_ratio <= 1.0):
            raise ValueError(f"capacity_ratio must be in (0, 1], got {capacity_ratio}")

        self.block = block
        self.dim = dim
        self.capacity_ratio = capacity_ratio
        self.aux_weight = float(aux_weight)

        self.router = nn.Linear(dim, 1, bias=False)
        ph = predictor_hidden if predictor_hidden is not None else dim
        self.predictor = nn.Sequential(
            nn.Linear(dim, ph), nn.GELU(), nn.Linear(ph, 1)
        )

    def _capacity(self, n: int) -> int:
        return max(1, min(n, int(self.capacity_ratio * n)))

    def predict_route(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token routing decision for causal inference (no future tokens).

        Returns a boolean ``(B, N)`` mask (True = send through the block).
        """
        return self.predictor(x).squeeze(-1) > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        C = self._capacity(N)

        r = self.router(x).squeeze(-1)                      # (B, N)
        topk_val, topk_idx = torch.topk(r, C, dim=1)        # (B, C)

        # Keep selected tokens in original order (helps a causal wrapped block).
        topk_idx, order = torch.sort(topk_idx, dim=1)
        topk_val = torch.gather(topk_val, 1, order)

        # Train the causal routing predictor (stop-gradient input, BCE target).
        if self.training and self.aux_weight != 0.0:
            threshold = topk_val[:, -1:].detach()
            target = (r >= threshold).to(r.dtype).detach()
            p_logits = self.predictor(x.detach()).squeeze(-1)
            bce = F.binary_cross_entropy_with_logits(p_logits, target)
            self.reg_loss = self.aux_weight * bce
        else:
            self.reg_loss = 0.0

        idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)      # (B, C, D)
        x_sel = torch.gather(x, 1, idx)                     # (B, C, D)
        y_sel = self.block(x_sel)                           # (B, C, D)
        y_sel = topk_val.unsqueeze(-1) * y_sel              # router-weighted

        out = x.clone()
        out.scatter_add_(1, idx, y_sel)                     # selected: r·f + x
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, capacity_ratio={self.capacity_ratio}"
