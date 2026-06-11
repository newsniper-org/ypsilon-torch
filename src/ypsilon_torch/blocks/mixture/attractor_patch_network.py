r"""Attractor Patch Networks (APN).

Reference
---------
*Attractor Patch Networks* (arXiv:2602.06993).

A sparse residual block: each token is matched (by cosine similarity) against
``K`` learned prototypes ("attractors"), the top-``k`` are selected, and each
selected patch adds a **low-rank** residual conditioned on a shared compact
code. A dense FFN is thereby replaced by a piecewise low-rank residual family.

.. math::

    s_i = \langle \mathrm{LN}(h), \widehat{p_i} \rangle / \tau, \qquad
    \mathcal{K} = \mathrm{TopK}_i(s_i, k), \qquad
    w_i = \mathrm{softmax}_{i\in\mathcal{K}}(s_i)

    u = V^\top \mathrm{LN}(h), \qquad
    \varphi_i = u \odot \sigma(a_i \odot u + b_i), \qquad
    \Delta_i = U_i \varphi_i

    y = h + \gamma \sum_{i\in\mathcal{K}} w_i\, \Delta_i

Note
----
Hyperparameter values from the source (``K=256``, ``r=32``, ``τ=0.07``,
``k∈[2,8]``) were extracted by an automated reader and flagged for
verification; the defaults below follow them but are easily overridden.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._aux_loss import AuxLossModule


class AttractorPatchNetwork(AuxLossModule):
    r"""Top-``k`` prototype-routed low-rank residual block.

    Parameters
    ----------
    dim : int
        Token dimension ``d`` (input == output).
    num_patches : int
        Number of prototypes / patch experts ``K``. Default ``256``.
    rank : int
        Compact-code dimension ``r`` (``r << d``). Default ``32``.
    top_k : int
        Number of active patches per token ``k``. Default ``4``.
    tau : float
        Routing temperature ``τ``. Default ``0.07``.
    gamma : float
        Residual scale ``γ``. Default ``1.0``.
    aux_weight : float
        Weight of the load-balancing loss exposed via the ``reg_loss`` property.
    """

    def __init__(
        self,
        dim: int,
        num_patches: int = 256,
        rank: int = 32,
        top_k: int = 4,
        tau: float = 0.07,
        gamma: float = 1.0,
        aux_weight: float = 1e-2,
    ) -> None:
        super().__init__()
        if top_k > num_patches:
            raise ValueError(f"top_k ({top_k}) must be <= num_patches ({num_patches})")

        self.dim = dim
        self.num_patches = num_patches
        self.rank = rank
        self.top_k = top_k
        self.tau = tau
        self.gamma = gamma
        self.aux_weight = float(aux_weight)

        self.norm = nn.LayerNorm(dim)
        self.prototypes = nn.Parameter(torch.empty(num_patches, dim))      # p_i
        self.code_proj = nn.Parameter(torch.empty(dim, rank))              # V
        self.gate_a = nn.Parameter(torch.zeros(num_patches, rank))         # a_i
        self.gate_b = nn.Parameter(torch.zeros(num_patches, rank))         # b_i
        self.decoder = nn.Parameter(torch.empty(num_patches, dim, rank))   # U_i

        nn.init.normal_(self.prototypes, std=dim ** -0.5)
        nn.init.normal_(self.code_proj, std=dim ** -0.5)
        nn.init.normal_(self.decoder, std=(dim * rank) ** -0.25)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.norm(h)                                          # (B, T, d)

        # Cosine routing to prototypes.
        zn = F.normalize(z, dim=-1)
        pn = F.normalize(self.prototypes, dim=-1)                 # (K, d)
        s = (zn @ pn.t()) / self.tau                             # (B, T, K)

        topv, topi = torch.topk(s, self.top_k, dim=-1)           # (B, T, k)
        w = F.softmax(topv, dim=-1)                              # (B, T, k)

        # Shared compact code and per-patch gated features.
        u = z @ self.code_proj                                   # (B, T, r)
        a_sel = self.gate_a[topi]                                # (B, T, k, r)
        b_sel = self.gate_b[topi]
        U_sel = self.decoder[topi]                               # (B, T, k, d, r)

        u_k = u.unsqueeze(2)                                     # (B, T, 1, r)
        phi = u_k * torch.sigmoid(a_sel * u_k + b_sel)          # (B, T, k, r)
        delta = torch.einsum("btkdr,btkr->btkd", U_sel, phi)    # (B, T, k, d)

        update = (w.unsqueeze(-1) * delta).sum(dim=2)            # (B, T, d)

        # Load-balancing loss: keep patch usage close to uniform 1/K.
        if self.training and self.aux_weight != 0.0:
            usage = torch.zeros(
                self.num_patches, dtype=h.dtype, device=h.device
            )
            counts = torch.bincount(
                topi.reshape(-1), minlength=self.num_patches
            ).to(h.dtype)
            usage = counts / counts.sum().clamp(min=1.0)
            l_bal = ((usage - 1.0 / self.num_patches) ** 2).sum()
            self.reg_loss = self.aux_weight * l_bal
        else:
            self.reg_loss = 0.0

        return h + self.gamma * update

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_patches={self.num_patches}, "
            f"rank={self.rank}, top_k={self.top_k}"
        )
