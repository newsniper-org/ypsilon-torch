# Bilinear MLP Block (AFBO-style Asymmetric Spatial-Channel Factorization)

> 한국어 버전: [bilinear_mlp.ko.md](bilinear_mlp.ko.md)

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

`ypsilon_torch.blocks.bilinear_mlp.AsymmetricSpatialChannelFactorizedBilinearMLP` is a
drop-in replacement for a **Vision Transformer FFN**. It fuses two recent
ideas:

- **Bilinear MLPs** (arXiv:2410.08417) — an MLP of the form
  `y = W_out ((W₁x) ⊙ (W₂x))` with **no element-wise nonlinearity** between
  the two linear projections. The whole layer is therefore expressible as a
  3rd-order tensor `B_{ijk}` in the input, which enables weight-only
  interpretability (eigendecomposition, circuit identification, etc.).
- **AFBO / SCFBO** (ICLR 2025) — factorizes each ViT-FFN linear projection
  into a **Spatial Modeling (SM)** component and a **Channel Mapping (CM)**
  component, and makes the two branches **asymmetric** by using different
  structured-sparsity channel strategies (GCCM, OCCM). This targets AFBO's
  performance / complexity trade-off.

This block keeps both branches linear (preserving the Bilinear-MLP property)
while decomposing each branch into `CM(SM(x))` (inheriting AFBO's
factorization), taking the benefits of both.

### Data flow

```
                 ┌──────────── Branch A ────────────┐
 x (B,N,C) ──┬──►│  SM_a  ─►  GCCM  ─►  a (B,N,H) │──┐
             │   └──────────────────────────────────┘  │
             │                                         ├─► a ⊙ b ─► Dropout ─► W_out ─► y (B,N,C_out)
             │   ┌──────────── Branch B ────────────┐  │
             └──►│  SM_b  ─►  OCCM  ─►  b (B,N,H) │──┘
                 └──────────────────────────────────┘
```

- `SM`: a **per-channel (depth-wise)** operator that acts along the token
  axis (dwconv / avg pool / identity).
- `GCCM`: **Grouped Cross Channel Mapping** — non-overlapping group
  partition + channel shuffle.
- `OCCM`: **Overlapped Cycle Channel Mapping** — cyclically overlapping
  group partition.
- The element-wise product of the two branches is the bilinear core; an
  output projection `W_out` sits on top.

---

## 2. Mathematical formulation

For an input `x ∈ ℝ^{B×N×C}`:

### Bilinear MLP core

$$
a = W_1 x, \qquad b = W_2 x, \qquad y = W_{\text{out}} (a \odot b).
$$

Because there is no element-wise nonlinearity, the whole operation is an
exact **third-order tensor** in the input:

$$
y_i \;=\; \sum_{j,k} B_{ijk}\, x_j\, x_k,
\qquad
B_{ijk} \;=\; \sum_{h} (W_{\text{out}})_{ih}\,(W_1)_{hj}\,(W_2)_{hk}.
$$

This is the structural basis of the weight-based interpretability claim of
Pearce et al.

### SCFBO factorization

Each linear branch is factorized as the composition of a **Spatial Modeling**
operator and a **Channel Mapping**:

$$
a \;=\; \mathrm{CM}_a\bigl(\mathrm{SM}_a(x)\bigr), \qquad
b \;=\; \mathrm{CM}_b\bigl(\mathrm{SM}_b(x)\bigr).
$$

- `SM`: per-channel along the token axis (length `N`). One of dwconv
  (`groups=C`), `AvgPool1d`, or identity.
- `CM`: a **structured-sparse linear** along the channel axis (`C → H`) —
  GCCM or OCCM (below).

### Asymmetry: GCCM vs OCCM

The two branches use **different** structured-sparsity channel mappings.
Let `G` be the number of groups.

**GCCM — Grouped Cross Channel Mapping**

- Partition the input channels into `G` non-overlapping groups of size
  `C/G`.
- Apply an independent linear `W_g ∈ ℝ^{(C/G)×(H/G)}` to each group.
- Follow with a **cyclic channel shuffle** (identical to ShuffleNet) so
  that subsequent layers — or the paired bilinear branch — mix
  information across groups.
- Parameter count: `G · (C/G) · (H/G) = C · H / G` — a `G`× saving over
  a full linear.

**OCCM — Overlapped Cycle Channel Mapping**

- Each output group reads from a **cyclically overlapping** slice of input
  channels. Group `g` sees channels
  `[g·(C/G) − s : g·(C/G) + (C/G) + s] mod C` (overlap `s`).
- Per-group weight: `W_g ∈ ℝ^{(C/G + 2s)×(H/G)}`.
- Parameter count: `G · (C/G + 2s) · (H/G) = C · H / G + 2s · H`.
- `s = 0` reduces OCCM to a pure grouped linear (no channel shuffle).

**Why asymmetric?** GCCM's channel shuffle produces a **global** cross-group
interaction (by interleaving outputs), whereas OCCM's cyclic overlap
introduces **local** cross-group connectivity directly at the weight level.
The two strategies are complementary, so the bilinear product `a ⊙ b` sees
two different kinds of cross-group information — this is the core of AFBO's
performance/complexity trade-off.

### Bilinearity identity (for sanity checks)

With all biases set to zero the block is **exactly homogeneous of degree 2**
in its input:

$$
\text{forward}(\alpha x) \;=\; \alpha^2 \,\text{forward}(x).
$$

Empirical check: the output ratio at `α = 2` should be `≈ 4.0` (see §6).

---

## 3. Architecture & API reference

### 3.1 `BilinearMLPBase(torch.nn.Module, abc.ABC)`

Abstract base for every bilinear-MLP block.

**Constructor**

```python
BilinearMLPBase(
    dim: int,
    hidden_dim: int | None = None,   # default: int(dim * expansion)
    out_dim: int | None = None,      # default: dim
    expansion: float = 4.0,
    dropout: float = 0.0,
    bias: bool = True,
)
```

| arg | meaning |
|---|---|
| `dim` | input channel size `C_in` |
| `hidden_dim` | bilinear-product dimension `H`. `None` → `dim * expansion` |
| `out_dim` | output channel size `C_out`. `None` → `dim` |
| `expansion` | used only when `hidden_dim is None` |
| `dropout` | dropout applied to `a ⊙ b` before `W_out` |
| `bias` | whether `out_proj` has a bias |

**Required overrides** (subclasses must implement)

```python
@abc.abstractmethod
def branch_a(self, x: Tensor) -> Tensor: ...   # (B,N,dim) → (B,N,hidden_dim)
@abc.abstractmethod
def branch_b(self, x: Tensor) -> Tensor: ...   # (B,N,dim) → (B,N,hidden_dim)
```

**`forward`**

```python
a = self.branch_a(x)
b = self.branch_b(x)
self._compute_aux_loss(a, b)        # subclass hook — updates reg_loss slot
h = self.dropout(a * b)
return self.out_proj(h)
```

**`reg_loss` property (torchutils.auxloss, requires `>= 0.1.1`)**

- Getter: returns the module's `_reg_loss` scalar tensor.
- Setter: `self.reg_loss = loss` accepts a `Tensor` or a Python scalar;
  internally normalizes dtype/shape.
- Collector (**static-style call**):
  `BilinearMLPBase.reg_loss.collect(model)` walks `model.modules()` and
  aggregates all `_reg_loss` entries via `torch.sum`.
- Resetter: `BilinearMLPBase.reg_loss.reset(model, 0.0)` walks `model.modules()`
  and clears every sub-module's `_reg_loss` to the scalar. (The plain
  `module.reg_loss = 0.0` setter only affects the single module it is called on,
  and only when that module subclasses `BilinearMLPBase`/`AuxLossModule` — it
  does **not** recurse, so use the resetter to clear a whole model.)

### 3.2 `AsymmetricSpatialChannelFactorizedBilinearMLP(BilinearMLPBase)`

The concrete AFBO-style block.

**Constructor**

```python
AsymmetricSpatialChannelFactorizedBilinearMLP(
    dim: int,
    hidden_dim: int | None = None,
    out_dim: int | None = None,
    expansion: float = 4.0,
    dropout: float = 0.0,
    bias: bool = True,
    sm_kind: Literal["dwconv", "pool", "none"] = "dwconv",
    sm_kernel_size: int = 3,
    num_groups: int = 8,
    occm_overlap: int = 1,
    aux_loss_weight: float = 0.0,
)
```

| arg | default | meaning · constraint |
|---|---|---|
| `sm_kind` | `"dwconv"` | spatial modeling operator. `"dwconv"` / `"pool"` / `"none"` |
| `sm_kernel_size` | `3` | `≥ 1` |
| `num_groups` | `8` | `G`. requires `dim % G == 0` and `hidden_dim % G == 0` |
| `occm_overlap` | `1` | `s`. `0 ≤ 2s ≤ dim / G` |
| `aux_loss_weight` | `0.0` | weight of the branch-decorrelation regularizer |

**Submodules**

- `sm_a`, `sm_b`: two independent `SpatialModeling` instances.
- `cm_a`: `GroupedCrossChannelMapping(dim, hidden_dim, num_groups)`
- `cm_b`: `OverlappedCycleChannelMapping(dim, hidden_dim, num_groups, occm_overlap)`

### 3.3 `SpatialModeling(nn.Module)`

```python
SpatialModeling(
    dim: int,
    kind: Literal["dwconv", "pool", "none"] = "dwconv",
    kernel_size: int = 3,
    bias: bool = False,
)
```

- Input/output: `(B, N, C)`.
- Internally transposes to `(B, C, N)` for Conv1d / AvgPool1d, then
  transposes back.
- Padding is `kernel_size // 2` (same padding).

### 3.4 `GroupedCrossChannelMapping(nn.Module)`

```python
GroupedCrossChannelMapping(
    in_channels: int,
    out_channels: int,
    num_groups: int,
    bias: bool = True,
)
```

- Weight shape: `(G, in_channels/G, out_channels/G)`.
- Forward: `einsum("...gi,gio->...go", x_g, W)` followed by a cyclic
  channel shuffle (transpose + reshape).
- Drop-in replacement for `nn.Linear(in_channels, out_channels)` on the
  last axis.

### 3.5 `OverlappedCycleChannelMapping(nn.Module)`

```python
OverlappedCycleChannelMapping(
    in_channels: int,
    out_channels: int,
    num_groups: int,
    overlap: int = 1,
    bias: bool = True,
)
```

- Weight shape: `(G, in_channels/G + 2·overlap, out_channels/G)`.
- `gather_idx` is a precomputed non-persistent buffer of shape
  `(G, window)` holding the cyclic input-channel indices each group reads.
- Forward: `x.index_select(-1, gather_idx.flatten())` → reshape →
  `einsum("...gw,gwo->...go", x_g, W)`.

---

## 4. Usage

### 4.1 Minimal example

```python
import torch
from ypsilon_torch.blocks.bilinear_mlp import AsymmetricSpatialChannelFactorizedBilinearMLP

block = AsymmetricSpatialChannelFactorizedBilinearMLP(
    dim=64, hidden_dim=256, num_groups=8,
).cuda()
x = torch.randn(2, 16, 64, device="cuda")
y = block(x)                         # (2, 16, 64)
```

### 4.2 ViT FFN replacement

```python
import torch
from torch import nn
from ypsilon_torch.blocks.bilinear_mlp import AsymmetricSpatialChannelFactorizedBilinearMLP

class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        # ---- Bilinear MLP instead of the standard FFN ----
        self.ffn = AsymmetricSpatialChannelFactorizedBilinearMLP(
            dim=dim,
            expansion=mlp_ratio,
            num_groups=8,
            occm_overlap=1,
            sm_kind="dwconv",
        )

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x
```

### 4.3 Integrating auxloss into a training loop

When constructed with `aux_loss_weight > 0.0`, every `forward` records the
cosine-square similarity between the two branches into the `reg_loss` slot.

```python
from ypsilon_torch.blocks.bilinear_mlp import BilinearMLPBase

model = MyViT(...).cuda()

for batch, target in loader:
    logits = model(batch)
    primary = criterion(logits, target)

    # Aggregate aux losses across every bilinear MLP block in the model
    aux = BilinearMLPBase.reg_loss.collect(model)

    loss = primary + aux           # aux is already scaled by aux_loss_weight
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Clear every submodule's _reg_loss for the next step. Use the resetter:
    # the `model.reg_loss = 0.0` setter does not recurse into submodules
    # (and `model` here is a plain nn.Module, not a BilinearMLPBase).
    BilinearMLPBase.reg_loss.reset(model, 0.0)
```

> `model` does **not** have to be a `BilinearMLPBase` instance. Both
> `reg_loss.collect` and `.reset` iterate over `model.modules()` and pick
> up any submodule that has a `_reg_loss` attribute.

### 4.4 Recovering the 3rd-order tensor (interpretability)

```python
import torch

# block.out_proj: Linear(H, C_out), H = hidden_dim
# branch_a / branch_b are linear, so their Jacobian == their weight matrix

B = 1 # batch
x0 = torch.zeros(B, 1, block.dim)
W_out = block.out_proj.weight          # (C_out, H)
# Effective linear matrix of each branch (∂a/∂x, ∂b/∂x)
W1 = torch.autograd.functional.jacobian(
    lambda x: block.branch_a(x.unsqueeze(0).unsqueeze(0))[0, 0], x0[0, 0]
)
W2 = torch.autograd.functional.jacobian(
    lambda x: block.branch_b(x.unsqueeze(0).unsqueeze(0))[0, 0], x0[0, 0]
)
# y_i = Σ_{j,k} B_{ijk} x_j x_k,  B_{ijk} = Σ_h W_out[i,h] W1[h,j] W2[h,k]
B_tensor = torch.einsum("ih,hj,hk->ijk", W_out, W1, W2)
# Eigendecomposition of B_tensor then enables Pearce-et-al. style analysis.
```

---

## 5. Hyperparameter guide

### `num_groups` (G)

| value | effect |
|---|---|
| `1` | GCCM/OCCM collapse to full linears (no asymmetry). Not recommended. |
| `4`–`8` | Balanced trade-off. Typical default. |
| `16`, `32` | `G`× parameter savings. Small models, lightweight inference. |

Constraint: `dim % G == 0`, `hidden_dim % G == 0`.

### `occm_overlap` (s)

| value | effect |
|---|---|
| `0` | OCCM degenerates to a pure grouped linear (GCCM without shuffle). |
| `1` (default) | 1-channel leakage with each cyclic neighbour. Recommended. |
| `≥ 2` | Stronger local cross-group connectivity. Respect `2s ≤ dim/G`. |

### `sm_kind` / `sm_kernel_size`

| `sm_kind` | properties |
|---|---|
| `"dwconv"` (default) | AFBO default. Learnable, most expressive. |
| `"pool"` | Zero parameters. Baseline / lightweight option. |
| `"none"` | Spatial modeling disabled. Close to a pure channel-mixing MLP. |

`sm_kernel_size` is usually `3`; setting it to `7` widens the neighbourhood.

### `aux_loss_weight`

| value | meaning |
|---|---|
| `0.0` (default) | Skip aux-loss computation entirely (writes `0` even in `train()` mode). The `reg_loss` slot stays live, so `collect`/`reset` still work. |
| `0.01`–`0.1` | Mild branch-decorrelation regularization. Typical range. |
| `> 0.1` | Strong regularization. Use only when the two branches tend to collapse. |

Note: in `eval()` mode the aux-loss computation is skipped even for positive
weights, to avoid inference overhead.

---

## 6. Implementation notes

### `_reg_loss` as a non-persistent buffer

```python
self.register_buffer("_reg_loss", torch.zeros(()), persistent=False)
```

- Automatically follows `model.to(device)` / `.to(dtype)`.
- `persistent=False` keeps it out of `state_dict`, so transient aux-loss
  values never leak into checkpoints.

### `torchutils >= 0.1.1` required

In v0.1.0, `auxloss.setter` went through `property.setter`, which calls
`type(self)(fget, fset, fdel, doc)`. But `auxloss.__init__` has the
**positional signature `(fget, fcollect, fset, freset, doc)`**, so the
user-provided setter was silently stored in `fcollect` and the actual
`fset` ended up as `None`. Attempting `self.reg_loss = x` then raised
`AttributeError: property 'reg_loss' has no setter`.

The v0.1.1 release (notes: *"fix setter bugs"*) rewrites `auxloss.setter`
to call `auxloss(fget, fcollect, fset=fset, freset, doc)` directly,
preserving `fset`. This block requires v0.1.1+, pinned via
`rev = "v0.1.1"` in `pyproject.toml`.

### Channel shuffle equals ShuffleNet's

The final step of `GroupedCrossChannelMapping`,

```python
y_g = y_g.transpose(-2, -1).contiguous().reshape(*lead, out_channels)
```

is exactly the channel-shuffle permutation of ShuffleNet v1/v2. It
interleaves per-group output channels so that the "next layer" sees a mix
of all groups — here, the "next layer" role is played by the other
bilinear branch.

### Bilinearity check (α² property)

With all biases zeroed the block is analytically 2-homogeneous:
`forward(αx) = α² forward(x)`. A CUDA sanity run at `α = 2` produces a mean
ratio `|y(2x) / y(x)|` of `4.0000`. If this identity breaks, a nonlinearity
has slipped in somewhere.

### Measured parameter savings

For `dim=64, hidden_dim=256, G=8, overlap=1`:

| module | #parameters |
|---|---|
| full `nn.Linear(64, 256)` + bias | 16 640 |
| GCCM | 2 304 (7.2× saving) |
| OCCM | 2 816 (5.9× saving) |

### CUDA performance

- The einsum paths (`"...gi,gio->...go"` etc.) are fused by PyTorch 2.10
  into cuBLAS grouped GEMMs, with negligible contiguity overhead in
  practice.
- OCCM's `index_select` gather is lightweight and does not bottleneck the
  forward/backward.

---

## 7. References

- **AFBO (ICLR 2025)**:
  "Asymmetric Factorized Bilinear Operation for Vision Transformer"
  — <https://openreview.net/forum?id=MJyqwBVgMs>,
  <https://iclr.cc/virtual/2025/poster/29941>
- **Bilinear MLPs (arXiv 2024)**:
  "Bilinear MLPs enable weight-based mechanistic interpretability"
  — <https://arxiv.org/abs/2410.08417>
- **ShuffleNet (CVPR 2018, v2: ECCV 2018)** — original source of the
  channel-shuffle operator.
- **`torchutils` / `auxloss` usage pattern**:
  <https://github.com/newsniper-org/torchutils>,
  <https://github.com/Honey-Be/framesmoothie> (reference usage).
