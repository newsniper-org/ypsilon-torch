"""Multi-Head FFN — a hardware-independent re-implementation of FlashMHF.

Reference
---------
Zhang, Hu, Li, Wu, Tu, *Flash Multi-Head FFN* (arXiv:2512.06989). The original
work reinterprets a SwiGLU FFN as a Multi-Head FFN with per-head, dynamically
weighted parallel sub-networks, and contributes a FlashAttention-style fused
kernel (Triton + an NVIDIA-Hopper/H100-only ThunderKittens variant using TMA /
WGMMA / warp-group specialization).

That fused kernel is **purely an I/O-aware optimisation**: it never
materialises the full ``L × d_ff`` intermediate, but it is *numerically
identical* to the plain formulation. The hardware lock-in (Hopper-specific
features) therefore lives entirely in the speed/memory layer, **not** in the
architecture.

This module reproduces the architecture (Eq. 10–14 of the paper) in pure
PyTorch with no kernel, custom op, or accelerator-specific dependency, so it
runs identically on CPU, consumer GPUs, and data-center GPUs. The optional
``memory_efficient`` path accumulates over sub-networks with a Python loop to
avoid building the ``L × d_ff`` tensor, recovering most of the memory benefit
(the speed benefit needs the fused kernel and is not reproduced here).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadFFN(nn.Module):
    r"""Multi-Head FFN with dynamically-weighted parallel sub-networks.

    For input ``X`` of shape ``(B, N, d_model)``:

    .. math::

        Q = \mathrm{reshape}(X W_{in},\ (B, N, H, d_h)), \quad d_h = d_{model}/H

        R^h = \mathrm{normalize}(\sigma(Q^h W_g^h)) \in \mathbb{R}^{B \times N \times E}

        S^h = \sum_{e=1}^{E} R^h_{e}\,
              \big(\mathrm{SiLU}(Q^h K_e^{h\top}) \odot (Q^h U_e^{h\top})\big) V_e^h

        O = \mathrm{concat}_H(S)\, W_{out}

    Parameters
    ----------
    dim : int
        Model dimension ``d_model`` (input == output). Must be divisible by
        ``num_heads``.
    num_heads : int
        Number of heads ``H``.
    num_subnetworks : int
        Parallel sub-networks per head ``E``.
    subnetwork_dim : int, optional
        Sub-network width ``d_e``. Defaults to ``round((8/3) * d_h)``
        (the SwiGLU-style expansion used by the paper).
    bias : bool
        Whether ``W_in`` / ``W_out`` use bias. Default ``False``.
    eps : float
        Stabiliser for the gate normalisation. Default ``1e-6``.
    memory_efficient : bool
        If True, accumulate over sub-networks with a Python loop, never
        building the full ``(B, N, H, E, d_e)`` intermediate. Default True.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_subnetworks: int = 8,
        subnetwork_dim: Optional[int] = None,
        bias: bool = False,
        eps: float = 1e-6,
        memory_efficient: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_subnetworks = num_subnetworks
        if subnetwork_dim is None:
            subnetwork_dim = max(1, int(round((8.0 / 3.0) * self.head_dim)))
        self.subnetwork_dim = subnetwork_dim
        self.eps = eps
        self.memory_efficient = memory_efficient

        H, E, dh, de = num_heads, num_subnetworks, self.head_dim, subnetwork_dim

        self.w_in = nn.Linear(dim, dim, bias=bias)
        self.w_out = nn.Linear(dim, dim, bias=bias)

        # Per-head gate: Q^h (d_h) -> E
        self.w_gate = nn.Parameter(torch.empty(H, dh, E))
        # Per (head, sub-network) projections K, U, V each (d_e, d_h).
        self.k = nn.Parameter(torch.empty(H, E, de, dh))
        self.u = nn.Parameter(torch.empty(H, E, de, dh))
        self.v = nn.Parameter(torch.empty(H, E, de, dh))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming-uniform style fan-in init for the structured parameters.
        for p in (self.k, self.u, self.v):
            nn.init.kaiming_uniform_(p.view(-1, p.shape[-1]), a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.head_dim)
        nn.init.uniform_(self.w_gate, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, E, dh = self.num_heads, self.num_subnetworks, self.head_dim

        q = self.w_in(x).reshape(B, N, H, dh)

        # Dynamic per-token sub-network weights R^h  (B, N, H, E).
        p = torch.einsum("bnhd,hde->bnhe", q, self.w_gate)
        r = torch.sigmoid(p)
        r = r / (r.sum(dim=-1, keepdim=True) + self.eps)

        if self.memory_efficient:
            s = torch.zeros_like(q)
            for e in range(E):
                qk = torch.einsum("bnhd,hfd->bnhf", q, self.k[:, e])  # (B,N,H,d_e)
                qu = torch.einsum("bnhd,hfd->bnhf", q, self.u[:, e])
                a = F.silu(qk) * qu
                contrib = torch.einsum("bnhf,hfd->bnhd", a, self.v[:, e])
                s = s + r[..., e : e + 1] * contrib
        else:
            qk = torch.einsum("bnhd,hefd->bnhef", q, self.k)  # (B,N,H,E,d_e)
            qu = torch.einsum("bnhd,hefd->bnhef", q, self.u)
            a = F.silu(qk) * qu
            contrib = torch.einsum("bnhef,hefd->bnhed", a, self.v)
            s = (r.unsqueeze(-1) * contrib).sum(dim=3)  # (B,N,H,d_h)

        o = s.reshape(B, N, self.dim)
        return self.w_out(o)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"num_subnetworks={self.num_subnetworks}, "
            f"subnetwork_dim={self.subnetwork_dim}"
        )
