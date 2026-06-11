# Feed-Forward (FFN) Replacement Blocks

> 한국어 버전: [feedforward.ko.md](feedforward.ko.md)

## Contents

1. [Overview](#1-overview)
2. [Mathematical formulation](#2-mathematical-formulation)
3. [Architecture & API reference](#3-architecture--api-reference)
4. [Usage](#4-usage)
5. [Hyperparameter guide](#5-hyperparameter-guide)
6. [Implementation notes](#6-implementation-notes)
7. [References](#7-references)

---

## 1. Overview

`ypsilon_torch.blocks.feedforward` provides two blocks that drop in for a
**Transformer / Vision Transformer FFN (Feed-Forward Network)**. Both are
`(B, N, d_model)` modules whose input and output dimensions match.

- **`HourglassFFN`** (arXiv:2602.06471 / 2510.01796) — where a standard FFN
  inflates once (`d_model → d_ff`, then back to `d_model`), this block instead
  **narrows** into a bottleneck `d_h < d_model` and stacks `K` residual
  sub-MLPs. It is an "hourglass" shape that reallocates the parameters usually
  spent on width to depth instead. Each sub-MLP is SwiGLU-style.
- **`MultiHeadFFN`** (arXiv:2512.06989 "Flash Multi-Head FFN") — reinterprets a
  SwiGLU FFN as a set of **per-head, dynamically weighted parallel
  sub-networks**. This implementation is a **hardware-independent pure-PyTorch
  re-implementation** of the original work.

### On `MultiHeadFFN`'s hardware lock-in

The original paper, arXiv:2512.06989, relies on a Triton kernel and — going
further — a FlashAttention-style fused kernel built on NVIDIA Hopper/H100-only
ThunderKittens (TMA / WGMMA / warp-group specialization). That kernel, however,
is **purely an I/O-aware optimization**: it merely avoids materializing the
full `L × d_ff` intermediate in memory and is **numerically identical** to the
plain pure-compute formulation. The hardware lock-in (Hopper-specific features)
therefore lives **only in the speed/memory layer, not in the architecture**.

This module reproduces the paper's architecture (Eq. 10–14) in pure PyTorch
with no kernel, custom op, or accelerator-specific dependency, so it runs
identically on CPUs, consumer GPUs, and data-center GPUs. The optional
`memory_efficient` path accumulates over sub-networks with a Python loop,
avoiding construction of the `L × d_ff` tensor and recovering most of the
memory benefit (the speed benefit needs the fused kernel and is not reproduced
here).

### Data flow

**HourglassFFN** — a residual sub-MLP stack that narrows into a bottleneck:

```
 x (B,N,d_model)
   │
   ├──────────────────────────────┐  (residual)
   ▼                              │
 norm ─► W_d1 ─► SiLU ─┐          │
   └──► W_d2 ──────────┤⊙─► W_u ──┤+──►  h₁
                       (d_h)      │
   ┆        ... × K (depth) ...   ┆
   ▼                              │
 h_K (B,N,d_model)
```

**MultiHeadFFN** — per-head, dynamically weighted parallel sub-networks:

```
 x (B,N,d_model)
   │
 W_in ─► reshape ─► Q (B,N,H,d_h)
   │
   ├─► gate:  R^h = normalize(σ(Q^h·W_g^h))   (B,N,H,E)
   │
   └─► head h, sub-network e=1..E:
          A_e = SiLU(Q^h·K_eᵀ) ⊙ (Q^h·U_eᵀ)        (d_e)
          S^h = Σ_e R^h_e · (A_e · V_e)             (d_h)
                              │
            concat_H(S) ─► W_out ─►  O (B,N,d_model)
```

---

## 2. Mathematical formulation

### 2.1 HourglassFFN

For an input `x ∈ ℝ^{B×N×d_model}`, set `h_0 = x` and pass through `K`
residual sub-MLPs:

$$
h_{i+1} \;=\; h_i + W_u^{(i)}\!\Big(
   \operatorname{SiLU}\!\big(W_{d1}^{(i)} \bar h_i\big)
   \;\odot\; \big(W_{d2}^{(i)} \bar h_i\big)\Big),
\qquad
\bar h_i = \operatorname{norm}(h_i).
$$

The output is `h_K`. Each sub-MLP is a **SwiGLU** gate built from two
down-projections `W_{d1}, W_{d2} \in \mathbb{R}^{d_h \times d_{model}}` and an
up-projection `W_u \in \mathbb{R}^{d_{model} \times d_h}`, with the intermediate
dimension being the bottleneck
`d_h = \mathrm{round}(d_{model}\cdot r) < d_{model}` (`r` is `bottleneck_ratio`).

Unlike a standard FFN that widens to `d_ff > d_model`, here we narrow to
`d_h < d_model` and invest the saved parameters in the depth `K`.

### 2.2 MultiHeadFFN

For an input `X ∈ ℝ^{B×N×d_model}`, first project with `W_in` and reshape into
`H` heads (`d_h = d_{model}/H`):

$$
Q = \mathrm{reshape}\big(X W_{in},\ (B, N, H, d_h)\big).
$$

The per-head dynamic gate `R^h \in \mathbb{R}^{B \times N \times E}` holds the
`E` sub-network weights, obtained via a sigmoid and then sum-normalized:

$$
R^h = \mathrm{normalize}\big(\sigma(Q^h W_g^h)\big),
\qquad
R^h_e = \frac{\sigma(Q^h W_g^h)_e}{\sum_{e'} \sigma(Q^h W_g^h)_{e'} + \varepsilon}.
$$

Each head output `S^h` is the dynamically weighted sum of the `E` sub-network
SwiGLU outputs:

$$
S^h \;=\; \sum_{e=1}^{E} R^h_{e}\;
   \Big(\operatorname{SiLU}\!\big(Q^h K_e^{h\top}\big)
   \odot \big(Q^h U_e^{h\top}\big)\Big) V_e^h,
$$

where each `(head, sub-network)` has
`K_e^h, U_e^h, V_e^h \in \mathbb{R}^{d_e \times d_h}`. Finally the heads are
concatenated and output-projected:

$$
O = \mathrm{concat}_H(S)\, W_{out}.
$$

`Q^h K_e^{h\top}` and `Q^h U_e^{h\top}` are the two branches of the SwiGLU gate
that lift into the sub-network width `d_e` (default `round((8/3)·d_h)`), and
`V_e^h` projects back down to `d_h`.

---

## 3. Architecture & API reference

### 3.1 `HourglassFFN(nn.Module)`

**Constructor**

```python
HourglassFFN(
    dim: int,
    bottleneck_ratio: float = 0.5,
    depth: int = 4,
    norm: Literal["layernorm", "rmsnorm", "none"] = "layernorm",
    bias: bool = False,
    hidden_dim: int | None = None,
)
```

| arg | default | meaning · constraint |
|---|---|---|
| `dim` | — | model dimension `d_model` (input == output). |
| `bottleneck_ratio` | `0.5` | `d_h / d_model`. Recommended `0.4`–`0.6` (the hourglass regime). `≥ 1` is allowed but defeats the purpose. |
| `depth` | `4` | number of stacked sub-MLPs `K`. Recommended `2`–`4`. Requires `≥ 1`. |
| `norm` | `"layernorm"` | per-sub-MLP pre-normalization. `"layernorm"` / `"rmsnorm"` / `"none"`. |
| `bias` | `False` | whether the linear layers use bias. |
| `hidden_dim` | `None` | explicit bottleneck size. When given, overrides `bottleneck_ratio`. |

**Submodules**

- `blocks`: an `nn.ModuleList` of `K` `_HourglassSubMLP` instances.
- each `_HourglassSubMLP`: `norm`, `w_d1` (`dim→hidden_dim`),
  `w_d2` (`dim→hidden_dim`), `w_u` (`hidden_dim→dim`).

**`forward`**

```python
h = x
for block in self.blocks:        # block(h) = w_u(SiLU(w_d1(norm(h))) * w_d2(norm(h)))
    h = h + block(h)
return h                          # (B, N, dim)
```

### 3.2 `MultiHeadFFN(nn.Module)`

**Constructor**

```python
MultiHeadFFN(
    dim: int,
    num_heads: int = 8,
    num_subnetworks: int = 8,
    subnetwork_dim: int | None = None,
    bias: bool = False,
    eps: float = 1e-6,
    memory_efficient: bool = True,
)
```

| arg | default | meaning · constraint |
|---|---|---|
| `dim` | — | model dimension `d_model` (input == output). Must be divisible by `num_heads`. |
| `num_heads` | `8` | number of heads `H`. `d_h = dim / H`. |
| `num_subnetworks` | `8` | parallel sub-networks per head `E`. |
| `subnetwork_dim` | `None` | sub-network width `d_e`. `None` → `round((8/3)·d_h)` (SwiGLU expansion). |
| `bias` | `False` | whether `W_in` / `W_out` use bias. |
| `eps` | `1e-6` | stabilizer `ε` for the gate normalization. |
| `memory_efficient` | `True` | if `True`, accumulate over `E` with a Python loop, never building the `(B,N,H,E,d_e)` intermediate. |

**Parameters**

- `w_in`, `w_out`: `nn.Linear(dim, dim)`.
- `w_gate`: an `nn.Parameter` of shape `(H, d_h, E)` — the per-head gate.
- `k`, `u`, `v`: each an `nn.Parameter` of shape `(H, E, d_e, d_h)` — the
  per-`(head, sub-network)` K/U/V projections.

**`forward`** (`memory_efficient=True` path)

```python
q = self.w_in(x).reshape(B, N, H, dh)
p = torch.einsum("bnhd,hde->bnhe", q, self.w_gate)
r = torch.sigmoid(p)
r = r / (r.sum(dim=-1, keepdim=True) + self.eps)

s = torch.zeros_like(q)
for e in range(E):
    qk = torch.einsum("bnhd,hfd->bnhf", q, self.k[:, e])   # (B,N,H,d_e)
    qu = torch.einsum("bnhd,hfd->bnhf", q, self.u[:, e])
    a = F.silu(qk) * qu
    contrib = torch.einsum("bnhf,hfd->bnhd", a, self.v[:, e])
    s = s + r[..., e : e + 1] * contrib

return self.w_out(s.reshape(B, N, self.dim))
```

---

## 4. Usage

> This project is managed with Poetry. Run/verify every snippet below with
> `poetry run python` (do not use plain `python`).

### 4.1 Minimal example

```python
import torch
from ypsilon_torch.blocks.feedforward import HourglassFFN, MultiHeadFFN

x = torch.randn(2, 16, 256)

hg = HourglassFFN(dim=256, bottleneck_ratio=0.5, depth=4)
y1 = hg(x)                           # (2, 16, 256)

mhf = MultiHeadFFN(dim=256, num_heads=8, num_subnetworks=8)
y2 = mhf(x)                          # (2, 16, 256)
```

### 4.2 ViT / Transformer FFN replacement (drop-in)

```python
import torch
from torch import nn
from ypsilon_torch.blocks.feedforward import HourglassFFN, MultiHeadFFN

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn: str = "hourglass"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        # ---- drop in where the standard FFN goes ----
        if ffn == "hourglass":
            self.ffn = HourglassFFN(dim=dim, bottleneck_ratio=0.5, depth=4)
        else:
            self.ffn = MultiHeadFFN(dim=dim, num_heads=num_heads, num_subnetworks=8)

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x
```

`HourglassFFN` carries its own internal residuals and norms, so it is fine to
use it under an outer `x + ffn(norm(x))` residual as above (the doubled
residual does not hurt training). It also works with the outer norm/residual
omitted, as just `x = self.ffn(x)`.

### 4.3 Checking `memory_efficient` path equivalence

```python
import torch
from ypsilon_torch.blocks.feedforward import MultiHeadFFN

x = torch.randn(2, 32, 256)

m = MultiHeadFFN(dim=256, num_heads=8, num_subnetworks=8, memory_efficient=True)
m.eval()
with torch.no_grad():
    y_loop = m(x)
    m.memory_efficient = False
    y_batched = m(x)

print((y_loop - y_batched).abs().max())   # ~1e-9 (numerically identical)
```

---

## 5. Hyperparameter guide

### HourglassFFN

#### `bottleneck_ratio` (r = d_h / d_model)

| value | effect |
|---|---|
| `0.4`–`0.6` (recommended) | the hourglass regime. Reallocate width savings into depth. |
| `≈ 1.0` | bottleneck effect vanishes. Close to a deep residual MLP. |
| `≥ 1.0` | allowed, but contradicts the hourglass intent (it widens). |

#### `depth` (K)

| value | effect |
|---|---|
| `1` | a single SwiGLU bottleneck sub-MLP. |
| `2`–`4` (recommended) | expressivity from depth. Typical range for replacing a standard FFN. |
| `> 4` | deeper, but more residual accumulation and compute. |

#### `norm`

| value | properties |
|---|---|
| `"layernorm"` (default) | stable default. |
| `"rmsnorm"` | common in LM-style stacks. No bias / mean subtraction, slightly lighter. |
| `"none"` | drops the pre-norm. Recommended only when norm is supplied externally. |

### MultiHeadFFN

#### `num_heads` (H) / `num_subnetworks` (E)

| arg | effect |
|---|---|
| `num_heads` | determines `d_h = dim/H`. Requires `dim % H == 0`. Larger H means smaller per-head dimension. |
| `num_subnetworks` | parallel experts per head. Larger means more expressivity, parameters, and compute. Around `8` is typical. |

#### `subnetwork_dim` (d_e)

| value | effect |
|---|---|
| `None` (default) | `round((8/3)·d_h)` — the paper's SwiGLU expansion ratio. |
| explicit | tune the sub-network's internal width directly. Smaller is lighter. |

#### `memory_efficient`

| value | meaning |
|---|---|
| `True` (default) | loop-accumulate over `E`. Never builds `(B,N,H,E,d_e)` → saves memory. Good for large `E` / long sequences. |
| `False` | batched einsum. Can be slightly faster for small `E` but uses more memory. Numerically identical. |

---

## 6. Implementation notes

### What removing the hardware lock-in means (MultiHeadFFN)

The performance contribution of arXiv:2512.06989 comes from a Hopper/H100-only
ThunderKittens (TMA/WGMMA/warp-group) fused kernel, but that kernel is just an
**I/O-aware optimization** that avoids materializing the `L × d_ff`
intermediate; its result is numerically identical to the plain formulation.
This re-implementation therefore **preserves the architecture (Eq. 10–14)
intact** and removes only the kernel dependency. As a result it runs identically
on CPUs, consumer GPUs, and data-center GPUs, and the only thing lost is the
fused kernel's speed benefit (most of the memory benefit is recovered by the
`memory_efficient` path).

### `memory_efficient` vs `batched` equivalence

Both paths compute the same formula.

```python
# memory_efficient=True:  loop-accumulate over E, never building (B,N,H,E,d_e)
for e in range(E):
    a = F.silu(q @ k[:, e].T) * (q @ u[:, e].T)
    s += r[..., e:e+1] * (a @ v[:, e])

# memory_efficient=False:  build (B,N,H,E,d_e) at once and sum over E
a = F.silu(einsum(q, k)) * einsum(q, u)         # (B,N,H,E,d_e)
s = (r.unsqueeze(-1) * einsum(a, v)).sum(dim=3) # (B,N,H,d_h)
```

They differ only in floating-point summation order, so the outputs are
numerically identical (verified, `max|Δ| ~ 1e-9`). The model behaves the same
whichever you pick for training or inference; the choice is purely a
memory/speed trade-off.

### HourglassFFN parameter efficiency

Each sub-MLP uses `W_{d1}, W_{d2} (d_model→d_h)` and `W_u (d_h→d_model)`, i.e.
about `3 · d_h · d_model` parameters per block. A `K`-deep stack therefore costs
roughly

$$
K \cdot 3 \cdot d_h \cdot d_{model}
$$

(excluding bias). Compared with a standard FFN (`2 · d_ff · d_model`, usually
`d_ff = 4·d_model`), the bottleneck `d_h < d_model` lets you spend the same
parameter budget on **depth instead of width**. For example, with
`d_model=256, r=0.5 (d_h=128), K=4`, each block is `≈ 3·128·256 = 98304`, and 4
of them `≈ 393216` — smaller than a `d_ff=1024` standard FFN
(`≈ 2·1024·256 = 524288`), while being 4× deep.

### Initialization

`MultiHeadFFN.reset_parameters` initializes the structured parameters `k`/`u`/`v`
with Kaiming-uniform (`a=sqrt(5)`) after flattening on the last axis, and
initializes `w_gate` from a uniform `±1/sqrt(d_h)`. This keeps the gate's
sigmoid input from saturating too aggressively.

---

## 7. References

- **Flash Multi-Head FFN (arXiv 2025)**:
  Zhang, Hu, Li, Wu, Tu, "Flash Multi-Head FFN"
  — <https://arxiv.org/abs/2512.06989>.
  This `MultiHeadFFN` is a hardware-independent pure-PyTorch re-implementation
  of the paper's architecture (Eq. 10–14).
- **Hourglass / shape convention**:
  Liao, Chen, Yi, Shiu, "Revisiting the Shape Convention of Transformer
  Language Models" — <https://arxiv.org/abs/2602.06471>;
  Chen, Lee, Liao, Shiu, "Rethinking the Shape Convention of an MLP"
  (OpenReview `bUtLHJn90a`) — <https://arxiv.org/abs/2510.01796>.
- **SwiGLU**: Shazeer, "GLU Variants Improve Transformer" (2020)
  — <https://arxiv.org/abs/2002.05202>. The gated activation shared by both
  blocks' sub-MLPs / sub-networks.
- **FlashAttention**: Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness" (2022)
  — <https://arxiv.org/abs/2205.14135>. The prototype of the I/O-aware design
  the MultiHeadFFN paper's fused kernel follows.
