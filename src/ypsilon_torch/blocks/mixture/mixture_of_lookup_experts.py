r"""Mixture-of-Lookup-Experts (MoLE).

Reference
---------
Jie, Tang, Han et al., *Mixture of Lookup Experts* (ICML 2025 Oral,
arXiv:2503.15798).

The key asymmetry: the **router** reads the contextual hidden state ``h``,
but each **routed expert** reads only the token *embedding* ``e`` (a function of
the token id). Because every routed expert's output therefore depends only on
the token id, after training the outputs ``FFN_j(Embedding(i))`` can be
**precomputed for every vocabulary id** into a lookup table (LUT) and the expert
FFNs discarded — at inference a routed expert is a table lookup, not a forward
pass. A shared expert keeps full contextual (``h``-dependent) capacity.

Routing is **dense** (softmax over all experts, no top-k), so no load-balancing
loss is required.

.. math::

    g = \mathrm{softmax}(h W_r^\top), \qquad
    h' = h + \mathrm{FFN_{shared}}(h) + \sum_{j=1}^{N} g_j\, \mathrm{FFN}_j(e)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._swiglu import SwiGLUFeedForward


class MixtureOfLookupExperts(nn.Module):
    r"""Dense routed experts over token embeddings + a shared contextual expert.

    Parameters
    ----------
    dim : int
        Model dimension.
    num_experts : int
        Number of routed (lookup) experts ``N``.
    routed_hidden : int, optional
        Routed-expert FFN width. Defaults to ``expansion * dim``.
    shared_hidden : int, optional
        Shared-expert FFN width. Defaults to ``expansion * dim``.
    expansion : float
        Width multiplier when explicit widths are not given. Default ``4.0``.
    use_shared : bool
        Whether to include the shared (contextual) expert. Default ``True``.
    bias : bool
        Whether FFN linear layers use bias. Default ``False``.

    Notes
    -----
    ``forward(h, e)`` takes the hidden state ``h`` and the token embedding ``e``
    (both ``(B, N, dim)``). For inference, call :meth:`build_lookup_table` once
    with the embedding matrix and then :meth:`forward_lookup` with token ids —
    this skips all routed-expert FFN computation.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        routed_hidden: Optional[int] = None,
        shared_hidden: Optional[int] = None,
        expansion: float = 4.0,
        use_shared: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.use_shared = use_shared

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUFeedForward(dim, hidden_dim=routed_hidden, expansion=expansion, bias=bias)
            for _ in range(num_experts)
        )
        self.shared_expert = (
            SwiGLUFeedForward(dim, hidden_dim=shared_hidden, expansion=expansion, bias=bias)
            if use_shared
            else None
        )

    def _routed(self, h: torch.Tensor, expert_out: torch.Tensor) -> torch.Tensor:
        """Combine router gates ``g(h)`` with per-expert outputs ``(.., N, dim)``."""
        g = F.softmax(self.router(h), dim=-1)                 # (B, N, num_experts)
        return torch.einsum("bne,bned->bnd", g, expert_out)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Training-mode forward. ``h``: hidden state, ``e``: token embedding."""
        expert_out = torch.stack([expert(e) for expert in self.experts], dim=2)
        out = h + self._routed(h, expert_out)
        if self.shared_expert is not None:
            out = out + self.shared_expert(h)
        return out

    @torch.no_grad()
    def build_lookup_table(self, embedding_weight: torch.Tensor) -> torch.Tensor:
        """Precompute the LUT ``(vocab_size, num_experts, dim)`` from embeddings.

        ``embedding_weight`` is the ``(vocab_size, dim)`` token-embedding matrix.
        Run once after training; store the result and pass it to
        :meth:`forward_lookup`.
        """
        # Stack each expert's output over the whole vocabulary.
        outs = [expert(embedding_weight) for expert in self.experts]  # (V, dim) each
        return torch.stack(outs, dim=1)                                # (V, N, dim)

    def forward_lookup(
        self, h: torch.Tensor, token_ids: torch.Tensor, lookup_table: torch.Tensor
    ) -> torch.Tensor:
        """Inference forward using a precomputed LUT (no routed-expert FFNs).

        Parameters
        ----------
        h : Tensor
            Hidden state ``(B, N, dim)``.
        token_ids : Tensor
            Integer token ids ``(B, N)``.
        lookup_table : Tensor
            Table from :meth:`build_lookup_table`, ``(vocab_size, N, dim)``.
        """
        expert_out = lookup_table[token_ids]                  # (B, N, num_experts, dim)
        out = h + self._routed(h, expert_out)
        if self.shared_expert is not None:
            out = out + self.shared_expert(h)
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_experts={self.num_experts}, use_shared={self.use_shared}"
