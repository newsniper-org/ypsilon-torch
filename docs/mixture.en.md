# Mixture (Sparse / Conditional-Computation) Blocks

> 한국어 버전: [mixture.ko.md](mixture.ko.md)

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

`ypsilon_torch.blocks.mixture` collects five **sparse / conditional-computation**
blocks. Each realizes the same idea — "don't spend the same compute on every
token" — along a different axis.

| Block | What is routed | Routing | aux loss |
|---|---|---|---|
| `MixtureOfDepths` (MoD) | whether a token enters the block (depth) | expert-choice top-k | predictor BCE |
| `MixtureOfDepthsAndExperts` (MoDE) | which expert (or no-op) a token goes to | token-choice top-1 | Switch load-balance |
| `MixtureOfHiddenDimensions` (MoHD) | which sub-dim groups of the hidden width | shared + top-k | Switch load-balance |
| `MixtureOfLookupExperts` (MoLE) | dense experts over token embeddings | dense softmax (no top-k) | none |
| `AttractorPatchNetwork` (APN) | prototype (attractor) patches | cosine top-k | usage uniform-balance |

Every block that emits an aux loss inherits
`ypsilon_torch.blocks.AuxLossModule`, which uses the **exact same** `reg_loss`
mechanism (backed by `torchutils.auxloss`) as `BilinearMLPBase` in the
[Bilinear MLP docs](bilinear_mlp.en.md). A **single**
`AuxLossModule.reg_loss.collect(model)` therefore gathers both the mixture
blocks and the bilinear blocks in a model (verified, §4.6).

### Data flow (one-glance per block)

```
MoD   x (B,N,d) ─► router r=w·x ─► top-C tokens only ─► f(X̃) ─► r·f + residual ─► (B,N,d)
                                  (remaining tokens bypass via residual)

MoDE  (integrated) x ─► token-choice top-1 router [E experts + 1 no-op] ─► x + g·expert(x)
      (staged)     x ─► MoD token selection ─► separate MoE router ─► expert(x)

MoHD  x ─► W_up ─► N sub-dim groups ─► [shared always-on | specialized top-k]
        ─► α scale ─► grouped fusion ─► SiLU ─► W_down ─► (B,N,d)

MoLE  router(h) ─┐
      e (embed) ─► FFN_j(e) ─► Σ g_j FFN_j(e)  ──► h + shared(h) + Σ ─► (B,N,d)
      (inference: precompute FFN_j(e) over the whole vocab into a LUT → index it)

APN   h ─► LN ─► cosine(K prototypes)/τ ─► top-k, softmax w
        ─► shared code u=Vᵀz, φ_i=u⊙σ(a_i⊙u+b_i), Δ_i=U_i φ_i
        ─► h + γ·Σ w_i Δ_i ─► (B,N,d)
```

---

## 2. Mathematical formulation

Notation: input `x ∈ ℝ^{B×N×d}` (batch `B`, sequence `N`, dimension `d`). `f`
is the wrapped block, `r_i` the router score of token `i`, `g` the gate
(softmax) weights.

### 2.1 Mixture-of-Depths (MoD)

The router produces a per-token scalar score `r_i = w_θ^⊤ x_i`. A capacity of
`C = ⌊capacity_ratio · N⌋` tokens is selected by **expert-choice top-k** (the
block picks the top-`C` tokens it will process). With the selected set
`X̃ = TopC_i(r_i)`:

$$
x_i^{l+1} = \begin{cases}
    r_i\, f(\tilde X)_i + x_i & r_i > P_\beta(R) \\
    x_i & \text{otherwise}
\end{cases}
\qquad \beta = 1 - C/N,
$$

where `P_β(R)` is the `β`-quantile of the score distribution `R` (the top-`C`
boundary). Only the selected tokens add `r_i·f(X̃)_i` to the residual; the rest
pass through via identity.

**Why no load-balance loss is needed.** Under expert-choice routing the block
selects exactly `C` tokens, so the load is structurally balanced — no
load-balancing loss (as required by token-choice MoE) is necessary.

**Predictor (for autoregressive inference).** Top-`k` over the whole sequence is
**non-causal** (it peeks at future tokens), so it cannot be used at generation
time. An auxiliary **predictor MLP** is therefore trained — with a
stop-gradient input — by BCE against top-`k` membership:

$$
\mathcal{L}_{\text{aux}}
= \mathrm{BCE}\bigl(\text{predictor}(\mathrm{sg}[x]),\;
  \mathbb{1}[r \ge \text{threshold}]\bigr),
\qquad
\text{threshold} = r_{(C)}.
$$

This `reg_loss = aux_weight · BCE` is added to the training loss; at inference,
`predict_route(x) = (predictor(x) > 0)` routes each token causally.

### 2.2 Mixture-of-Depths-and-Experts (MoDE)

arXiv:2404.02258 §4. Combines MoD's depth routing with a sparse MoE. Two
variants:

**integrated (paper-preferred).** Adds one **no-op (identity) expert** to the
`N` real experts and uses a **single token-choice top-1 router** over the
`N+1` slots. Routing a token to the no-op slot is exactly an MoD bypass, so
depth selection and expert selection share one routing op. With
`g = softmax(W_r x)` and `(g*, e*) = max_e g`:

$$
y_i = x_i + g^*_i \cdot \mathrm{FFN}_{e^*_i}(x_i)
\quad (\mathrm{FFN} = 0 \text{ for the no-op slot}).
$$

**staged.** An MoD router first selects which tokens are processed at all
(§2.1), then a separate MoE router assigns each surviving token to an expert
(`residual=False`).

**Switch-style load-balance loss** (over all slots, no-op included):

$$
\mathcal{L}_{\text{bal}}
= \alpha \cdot S \cdot \sum_{s=1}^{S} f_s\, p_s,
\qquad
f_s = \tfrac{1}{BN}\!\sum_{i} \mathbb{1}[e^*_i = s],
\quad
p_s = \tfrac{1}{BN}\!\sum_{i} g_{i,s},
$$

where `S` is the number of slots (`N` or `N+1`), `f_s` the fraction of tokens
dispatched to slot `s`, `p_s` the mean routing probability of slot `s`, and
`α = aux_weight`.

### 2.3 Mixture-of-Hidden-Dimensions (MoHD)

arXiv:2412.05644. Where MoE splits an FFN into expert *sub-networks*, MoHD
splits the **hidden dimension** `d_ff` into `N` sub-dim groups (each of size
`d_e = d_ff/N`) and activates only some per token. The router scores affinity
from centroids `φ_i`:

$$
s = \mathrm{softmax}(x\, C^\top) \in \mathbb{R}^{N_{\text{sub}}},
\qquad C \in \mathbb{R}^{N_{\text{sub}}\times d}.
$$

The first `shared` groups are **always on**; among the remaining `specialized`
groups, `top-k` are selected (`k = active − shared`). The gate `g` passes `s`
through for shared groups and keeps `s` only at the top-k positions for
specialized groups.

The activation magnitude lost to gating is restored by a **per-token scale
`α`**:

$$
\alpha = \Bigl(\sum_i g_i\Bigr)\cdot N_{\text{sub}},
\qquad
h = \alpha \cdot \bigl(g \odot (W_{\text{up}} x)\bigr).
$$

A **grouped fusion** (block-diagonal linear) then re-mixes within groups,
followed by SiLU and the down-projection:

$$
y = W_{\text{down}}\,\mathrm{SiLU}\bigl(\mathrm{Fusion}(h)\bigr).
$$

> The grouped fusion is **not** the paper's exact **Monarch matrix** (`O(d/r)`
> cost) but a **block-diagonal grouped linear** playing the same cheap,
> structured re-mixing role. Set `fusion=False` to drop it.

**load-balance loss** (Switch-style over specialized groups): with `f` the
per-group selection frequency, `p` the per-group mean affinity, and `n_spec`
the number of specialized groups, `L = α · n_spec · Σ f_g p_g`.

### 2.4 Mixture-of-Lookup-Experts (MoLE)

arXiv:2503.15798. **The key asymmetry** — the router reads the contextual hidden
state `h`, but each **routed expert reads only the token embedding `e`** (a
function of the token id). Routing is **dense** (softmax over all experts, no
top-k), so no load-balance loss is needed:

$$
g = \mathrm{softmax}(h\, W_r^\top), \qquad
h' = h + \mathrm{FFN}_{\text{shared}}(h) + \sum_{j=1}^{N} g_j\, \mathrm{FFN}_j(e).
$$

The shared expert depends on `h` and keeps full contextual capacity.

**LUT re-parameterization (inference).** Each routed expert's output `FFN_j(e)`
is a function of the token id, so after training it can be precomputed **over
the whole vocabulary** into a LUT `∈ ℝ^{V×N×d}`:

$$
\mathrm{LUT}[v, j, :] = \mathrm{FFN}_j\bigl(\mathrm{Embedding}(v)\bigr).
$$

At inference no expert FFN runs at all — the LUT is indexed by token id:

$$
h' = h + \mathrm{FFN}_{\text{shared}}(h)
     + \sum_j g_j\, \mathrm{LUT}[\text{id}, j, :].
$$

This is **not** an approximation — it is **exactly identical** (verified:
train vs LUT maxdiff `0.0`, §6).

### 2.5 Attractor Patch Networks (APN)

arXiv:2602.06993. Replaces a dense FFN with a family of **piecewise low-rank
residuals**. The LN'd token `z = LN(h)` is scored by cosine similarity against
`K` prototypes (attractors) `p_i`, scaled by temperature `τ`; the top-`k` are
selected and softmax-weighted:

$$
s_i = \frac{\langle \mathrm{LN}(h),\, \widehat{p_i}\rangle}{\tau},
\qquad
\mathcal{K} = \mathrm{TopK}_i(s_i, k),
\qquad
w_i = \mathrm{softmax}_{i\in\mathcal{K}}(s_i).
$$

From a **compact code** `u` shared by all patches (dimension `r ≪ d`), a
per-patch gate produces a low-rank residual `Δ_i`:

$$
u = V^\top \mathrm{LN}(h) \in \mathbb{R}^r,
\qquad
\varphi_i = u \odot \sigma(a_i \odot u + b_i),
\qquad
\Delta_i = U_i\, \varphi_i,
$$

$$
y = h + \gamma \sum_{i\in\mathcal{K}} w_i\, \Delta_i.
$$

**load-balance loss.** Keep patch usage close to the uniform `1/K`:

$$
\mathcal{L}_{\text{bal}}
= \alpha \sum_{i=1}^{K} \Bigl(\mathrm{usage}_i - \tfrac{1}{K}\Bigr)^2,
\qquad
\mathrm{usage}_i = \frac{\#\{i \text{ chosen into top-k}\}}{\sum_j \#\{\cdots\}}.
$$

> **Caution:** the default hyperparameters (`K=256`, `r=32`, `τ=0.07`,
> `k∈[2,8]`) were extracted from the source by an automated reader and are
> flagged for verification. They run as-is, but cross-check them against the
> original paper before production use.

---

## 3. Architecture & API reference

### 3.0 `AuxLossModule(torch.nn.Module)`

Base for every mixture block that emits an aux loss. A standalone base placing
the **same** `reg_loss` mechanism as `BilinearMLPBase` on top of
`torchutils.auxloss`.

- Getter: `self.reg_loss` → the module's `_reg_loss` scalar tensor.
- Setter: `self.reg_loss = loss` (a `Tensor` or a Python scalar); normalizes
  dtype/shape.
- Collector (**static-style call**): `AuxLossModule.reg_loss.collect(model)`
  walks `model.modules()` and aggregates every `_reg_loss` via `torch.sum` —
  gathering mixture **and** bilinear blocks together.
- Resetter: `AuxLossModule.reg_loss.reset(model, 0.0)` or `model.reg_loss = 0.0`.

`_reg_loss` is a non-persistent buffer: it follows `.to(device/dtype)` but
stays out of `state_dict`.

### 3.1 `MixtureOfDepths(AuxLossModule)`

```python
MixtureOfDepths(
    block: nn.Module,
    dim: int,
    capacity_ratio: float = 0.125,
    aux_weight: float = 1.0,
    predictor_hidden: int | None = None,
)
```

| arg | default | meaning |
|---|---|---|
| `block` | — | wrapped block, `(B, C, d) → (B, C, d)`. Only sees selected tokens. |
| `dim` | — | token dimension `d` |
| `capacity_ratio` | `0.125` | fraction `C/N` of tokens processed (paper's 12.5%) |
| `aux_weight` | `1.0` | predictor BCE weight. `0.0` disables only the loss contribution (predictor still usable at inference) |
| `predictor_hidden` | `None`(=`dim`) | hidden width of the causal predictor MLP |

- `forward(x) -> (B, N, d)`: expert-choice top-`C` routing. Selected tokens are
  re-sorted to original order (for causal wrapped blocks). The output is added
  to the residual via `scatter_add_`.
- `predict_route(x) -> bool (B, N)`: causal routing mask for autoregressive
  inference (`predictor(x) > 0`).
- Submodules: `router = nn.Linear(dim, 1, bias=False)`, `predictor =
  Sequential(Linear(dim,ph), GELU, Linear(ph,1))`.

### 3.2 `MixtureOfDepthsAndExperts(AuxLossModule)`

```python
MixtureOfDepthsAndExperts(
    dim: int,
    num_experts: int = 4,
    variant: Literal["integrated", "staged"] = "integrated",
    capacity_ratio: float = 0.125,
    expert_hidden: int | None = None,
    expansion: float = 4.0,
    aux_weight: float = 1e-2,
)
```

| arg | default | meaning |
|---|---|---|
| `dim` | — | token dimension |
| `num_experts` | `4` | number of real experts `N` (integrated adds 1 no-op slot) |
| `variant` | `"integrated"` | `"integrated"` (paper-preferred) / `"staged"` |
| `capacity_ratio` | `0.125` | token fraction kept by the MoD stage (`"staged"` only) |
| `expert_hidden` | `None`(=`expansion·dim`) | expert FFN width |
| `expansion` | `4.0` | width multiplier when `expert_hidden` is not given |
| `aux_weight` | `1e-2` | weight of the MoE load-balance (and the staged MoD predictor) |

- Experts are `SwiGLUFeedForward(dim, ...)`. integrated holds one `_Top1MoE`
  (with no-op, residual); staged wraps a `_Top1MoE` (no no-op, no residual) in
  a `MixtureOfDepths`.

### 3.3 `MixtureOfHiddenDimensions(AuxLossModule)`

```python
MixtureOfHiddenDimensions(
    dim: int,
    hidden_dim: int | None = None,
    expansion: float = 4.0,
    num_subdims: int = 16,
    active_ratio: float = 0.5,
    shared_ratio: float = 0.375,
    fusion: bool = True,
    aux_weight: float = 1e-2,
    bias: bool = False,
)
```

| arg | default | meaning · constraint |
|---|---|---|
| `dim` | — | model dimension (input == output) |
| `hidden_dim` | `None`(=`expansion·dim`) | intermediate width `d_ff`. Must be divisible by `num_subdims` |
| `num_subdims` | `16` | number of sub-dim groups `N` (paper's best) |
| `active_ratio` | `0.5` | fraction `δ` of groups active per token (shared + specialized) |
| `shared_ratio` | `0.375` | always-on shared fraction `φ`. `0 ≤ φ ≤ δ` (≈3/4 of active) |
| `fusion` | `True` | whether to apply the grouped fusion |
| `aux_weight` | `1e-2` | load-balance weight |
| `bias` | `False` | projection bias |

- Submodules: `w_up = Linear(dim, hidden_dim)`, `w_down = Linear(hidden_dim, dim)`,
  a `centroids` parameter `(num_subdims, dim)`, and `fusion =
  _GroupedFusion(hidden_dim, groups=num_subdims)` (block-diagonal linear).
- Internal counts: `n_shared = round(φ·N)`, `n_active = max(n_shared,
  round(δ·N))`, `n_specialized = n_active − n_shared`.

### 3.4 `MixtureOfLookupExperts(nn.Module)`

> Because it emits no aux loss (dense routing), it inherits plain `nn.Module`,
> not `AuxLossModule`.

```python
MixtureOfLookupExperts(
    dim: int,
    num_experts: int = 4,
    routed_hidden: int | None = None,
    shared_hidden: int | None = None,
    expansion: float = 4.0,
    use_shared: bool = True,
    bias: bool = False,
)
```

| arg | default | meaning |
|---|---|---|
| `dim` | — | model dimension |
| `num_experts` | `4` | number of routed (lookup) experts `N` |
| `routed_hidden` | `None`(=`expansion·dim`) | routed-expert FFN width |
| `shared_hidden` | `None`(=`expansion·dim`) | shared-expert FFN width |
| `expansion` | `4.0` | width multiplier when widths are not given |
| `use_shared` | `True` | include the shared (contextual) expert |
| `bias` | `False` | FFN linear bias |

- `forward(h, e) -> (B, N, d)`: training. `h` hidden state, `e` token embedding.
- `build_lookup_table(embedding_weight) -> (V, N, d)`: run once after training to
  precompute the LUT.
- `forward_lookup(h, token_ids, lookup_table) -> (B, N, d)`: inference. Indexes
  `lookup_table[token_ids]` with no routed-expert FFN computation.

### 3.5 `AttractorPatchNetwork(AuxLossModule)`

```python
AttractorPatchNetwork(
    dim: int,
    num_patches: int = 256,
    rank: int = 32,
    top_k: int = 4,
    tau: float = 0.07,
    gamma: float = 1.0,
    aux_weight: float = 1e-2,
)
```

| arg | default | meaning · constraint |
|---|---|---|
| `dim` | — | token dimension `d` (input == output) |
| `num_patches` | `256` | number of prototypes / patches `K` |
| `rank` | `32` | compact-code dimension `r` (`r ≪ d`) |
| `top_k` | `4` | active patches per token `k`. `k ≤ K` |
| `tau` | `0.07` | routing temperature `τ` |
| `gamma` | `1.0` | residual scale `γ` |
| `aux_weight` | `1e-2` | load-balance weight |

- Parameters: `prototypes (K, d)`, `code_proj V (d, r)`, `gate_a/gate_b (K, r)`,
  `decoder U (K, d, r)`, `norm = LayerNorm(dim)`.
- `forward(h) -> (B, T, d)`: cosine top-k routing → shared code → sum of
  low-rank residuals.

---

## 4. Usage

> This project is managed with **poetry**. Every example below was verified
> with `poetry run python` (plain `python` is not used).
> Imports: `from ypsilon_torch.blocks.mixture import (...)`,
> `from ypsilon_torch.blocks import AuxLossModule`.

### 4.1 MixtureOfDepths — wrapping an FFN

```python
import torch
from torch import nn
from ypsilon_torch.blocks.mixture import MixtureOfDepths

ffn = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))
mod = MixtureOfDepths(block=ffn, dim=64, capacity_ratio=0.125).cuda()

x = torch.randn(2, 128, 64, device="cuda")
y = mod(x)                              # (2, 128, 64) — only 12.5% of tokens enter ffn

# causal routing mask for autoregressive inference
mod.eval()
route = mod.predict_route(x)            # (2, 128) bool
```

### 4.2 MixtureOfDepthsAndExperts — integrated / staged

```python
import torch
from ypsilon_torch.blocks.mixture import MixtureOfDepthsAndExperts

x = torch.randn(2, 128, 64, device="cuda")

# paper-preferred integrated: experts + no-op, single top-1 router
mode = MixtureOfDepthsAndExperts(
    dim=64, num_experts=4, variant="integrated",
).cuda()
y = mode(x)                             # (2, 128, 64)

# staged: MoD selection then a separate MoE
mode_s = MixtureOfDepthsAndExperts(
    dim=64, num_experts=4, variant="staged", capacity_ratio=0.25,
).cuda()
y_s = mode_s(x)                         # (2, 128, 64)
```

### 4.3 MixtureOfHiddenDimensions — FFN replacement

```python
import torch
from ypsilon_torch.blocks.mixture import MixtureOfHiddenDimensions

mohd = MixtureOfHiddenDimensions(
    dim=64, hidden_dim=256, num_subdims=16,
    active_ratio=0.5, shared_ratio=0.375,
).cuda()
x = torch.randn(2, 128, 64, device="cuda")
y = mohd(x)                             # (2, 128, 64)
```

### 4.4 MixtureOfLookupExperts — LUT inference after training

```python
import torch
from torch import nn
from ypsilon_torch.blocks.mixture import MixtureOfLookupExperts

dim, vocab = 64, 100
emb = nn.Embedding(vocab, dim).cuda()
mole = MixtureOfLookupExperts(dim=dim, num_experts=4).cuda()

# --- training: router reads h, routed experts read embedding e ---
ids = torch.randint(0, vocab, (2, 16), device="cuda")
h = torch.randn(2, 16, dim, device="cuda")
e = emb(ids)
y = mole(h, e)                          # (2, 16, 64)

# --- inference: precompute the LUT, then index it (no expert FFNs) ---
mole.eval()
with torch.no_grad():
    lut = mole.build_lookup_table(emb.weight)   # (vocab, N, dim)
    y_lut = mole.forward_lookup(h, ids, lut)    # (2, 16, 64)
    # not an approximation — exactly the training path
    assert torch.allclose(mole(h, emb(ids)), y_lut, atol=1e-5)
```

### 4.5 AttractorPatchNetwork — residual block

```python
import torch
from ypsilon_torch.blocks.mixture import AttractorPatchNetwork

apn = AttractorPatchNetwork(
    dim=64, num_patches=256, rank=32, top_k=4, tau=0.07,
).cuda()
h = torch.randn(2, 16, 64, device="cuda")
y = apn(h)                              # (2, 16, 64) — h + γ·Σ w_i Δ_i
```

### 4.6 Integrating auxloss into a training loop (mixture + bilinear together)

Both mixture and bilinear blocks use `AuxLossModule`'s `reg_loss` mechanism, so
a single `collect` gathers them all.

```python
import torch
from torch import nn
from ypsilon_torch.blocks import AuxLossModule
from ypsilon_torch.blocks.mixture import MixtureOfHiddenDimensions
from ypsilon_torch.blocks.bilinear_mlp import (
    AsymmetricSpatialChannelFactorizedBilinearMLP,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mohd = MixtureOfHiddenDimensions(dim=64, hidden_dim=256)
        self.afbo = AsymmetricSpatialChannelFactorizedBilinearMLP(
            dim=64, hidden_dim=256, num_groups=8, aux_loss_weight=0.1,
        )
    def forward(self, x):
        return self.afbo(self.mohd(x))

model = Model().cuda().train()

for batch, target in loader:
    logits = model(batch)
    primary = criterion(logits, target)

    # Sum the aux losses of the mixture + bilinear blocks in one call
    aux = AuxLossModule.reg_loss.collect(model)

    loss = primary + aux           # aux is already scaled by each aux_weight
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Clear every submodule's _reg_loss for the next step
    AuxLossModule.reg_loss.reset(model, 0.0)
    # if model itself subclasses AuxLossModule, model.reg_loss = 0.0 also works
```

> `model` does **not** have to be an `AuxLossModule` instance. Both `collect`
> and `reset` walk `model.modules()` and pick up every submodule that has a
> `_reg_loss` attribute. Note, however, that the `model.reg_loss = 0.0`
> **setter syntax** works only when `model` itself subclasses `AuxLossModule`;
> for a plain `nn.Module` (as above) use `AuxLossModule.reg_loss.reset(model,
> 0.0)` to clear every submodule.

---

## 5. Hyperparameter guide

### MoD: `capacity_ratio`

| value | effect |
|---|---|
| `0.125` (default) | paper's 12.5%. Large reduction in average per-token compute. |
| `0.25`–`0.5` | conservative. Smaller accuracy loss, smaller savings. |
| `1.0` | all tokens processed. MoD disabled (behaves like the plain block). |

`aux_weight`: keep `> 0` (default `1.0`) to train the predictor. If you only
train with top-k and never do autoregressive inference, `0.0` is fine (the
predictor stays usable).

### MoDE: `variant` / `num_experts`

| `variant` | properties |
|---|---|
| `"integrated"` (default) | paper-preferred. Depth and expert choice share one router; the no-op slot acts as the MoD bypass. |
| `"staged"` | MoD selection then a separate MoE. `capacity_ratio` controls throughput directly. |

`num_experts` is usually `4`–`8`. Following the Switch convention, `aux_weight =
1e-2` is recommended (too large and the router fixates on uniform dispatch,
weakening specialization).

### MoHD: `num_subdims` / `active_ratio` / `shared_ratio`

| arg | recommended | note |
|---|---|---|
| `num_subdims` | `16` | paper's best. Must divide `hidden_dim`. |
| `active_ratio` (δ) | `0.5` | half the groups active per token. |
| `shared_ratio` (φ) | `0.375` | ≈3/4 of active is shared. `0 ≤ φ ≤ δ`. |
| `fusion` | `True` | `False` drops grouped fusion, lighter. |

Setting `shared_ratio == active_ratio` leaves zero specialized groups, which
removes routing entirely (all-shared ≈ a dense FFN).

### MoLE: `num_experts` / `use_shared`

| arg | recommended | note |
|---|---|---|
| `num_experts` | `4`–`N` | dense routing, so the count directly drives the LUT size `V×N×d`. |
| `use_shared` | `True` | the shared expert carries the `h` context capacity. Off → pure lookup. |

LUT memory: `V × num_experts × dim`. A large vocab plus many experts makes the
LUT big, so tune `num_experts` to control inference memory.

### APN: `num_patches` / `rank` / `top_k` / `tau`

| arg | default | note |
|---|---|---|
| `num_patches` (K) | `256` | number of patches. ⚠ verify-needed default (§2.5). |
| `rank` (r) | `32` | compact-code dimension. `r ≪ d`. ⚠ verify-needed. |
| `top_k` (k) | `4` | active patches per token. Paper range `[2, 8]`. |
| `tau` (τ) | `0.07` | routing temperature. Smaller = sharper. ⚠ verify-needed. |
| `gamma` (γ) | `1.0` | residual scale. |

`aux_weight = 1e-2` is recommended; too small and usage can collapse onto a few
patches.

---

## 6. Implementation notes

### Shared `reg_loss` (same mechanism as bilinear, verified)

The mixture blocks' `AuxLossModule` uses the same `torchutils.auxloss`-backed
`reg_loss` as `BilinearMLPBase`. Even with the two mixed in one model, a single
`AuxLossModule.reg_loss.collect(model)` gathers both losses. Verification (the
`§4.6` structure): the combined collect returns a non-zero scalar, and becomes
`0.0` immediately after `reset`.

`_reg_loss` is a `register_buffer(..., persistent=False)`: it follows
`.to(device/dtype)` but never enters `state_dict`, so transient aux values do
not leak into checkpoints.

### Aux loss skipped in `eval()` / when `aux_weight==0`

MoD/MoDE/MoHD/APN all compute the aux loss only when `self.training and
self.aux_weight != 0.0`; otherwise they write `self.reg_loss = 0.0`. This avoids
inference overhead. `collect`/`reset` still work (the slot is always live).

### MoD: expert-choice → no load-balance loss

MoD is **expert-choice** (the block picks the top-`C` tokens), not token-choice
(tokens picking experts). In this direction throughput is fixed at exactly `C`,
so the load is structurally balanced — hence no load-balance loss, only the
predictor BCE for causal inference.

The top-`C` indices are restored to original order via `torch.sort`, then
`gather` → `block` → `scatter_add_`. The sort preserves token order for a
causal wrapped block.

### MoLE: the LUT is exact, not an approximation

Because a routed expert reads only the embedding `e`, `FFN_j(Embedding(v))` is a
pure function of token id `v`. `build_lookup_table` evaluates exactly this over
the whole vocabulary and stores it; `forward_lookup` merely replaces the same
FFN with an index lookup. The training path and the LUT path are therefore
**numerically identical** — maxdiff `0.0` in verification.

### MoHD: grouped fusion is a lightweight Monarch substitute

The paper uses a **Monarch matrix** (`O(d/r)` cost) for grouped fusion. This
implementation substitutes a **block-diagonal grouped linear**
(`einsum("...gb,gbc->...gc", xg, weight)`) playing the same cheap
structured-re-mixing role — mixing only within groups, with no Monarch
permutation. `fusion=False` turns it off entirely.

The `α = (Σ g_i)·N_sub` scale compensates for the magnitude lost to gated-off
groups, keeping the down-projection input statistics close to a dense FFN.

### APN: verify the default hyperparameters

`K=256`, `r=32`, `τ=0.07`, `k∈[2,8]` were extracted from the source by an
automated reader and carry an explicit **verification flag** in the docstring.
They run as-is, but cross-check them against the original paper before
production (both this doc and the source state this).

The low-rank residual is computed per patch as `U_i φ_i` via
`einsum("btkdr,btkr->btkd", U_sel, phi)`. `U_sel = decoder[topi]` is gathered to
shape `(B, T, k, d, r)`.

---

## 7. References

- **Mixture-of-Depths (MoD / MoDE)**:
  Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro,
  "Mixture-of-Depths: Dynamically allocating compute in transformer-based
  language models" — <https://arxiv.org/abs/2404.02258> (MoDE is §4).
- **Mixture-of-Hidden-Dimensions (MoHD)**:
  Chen et al., "Mixture of Hidden-Dimensions: Not All Hidden-States'
  Dimensions are Needed in Transformer" (ICML 2025)
  — <https://arxiv.org/abs/2412.05644>.
- **Mixture-of-Lookup-Experts (MoLE)**:
  Jie, Tang, Han et al., "Mixture of Lookup Experts" (ICML 2025 Oral)
  — <https://arxiv.org/abs/2503.15798>.
- **Attractor Patch Networks (APN)**:
  "Attractor Patch Networks" — <https://arxiv.org/abs/2602.06993>.
  (Default hyperparameters are auto-extracted — verify against the paper.)
- **Switch Transformer** — source of the load-balancing loss:
  Fedus, Zoph, Shazeer, "Switch Transformers: Scaling to Trillion Parameter
  Models with Simple and Efficient Sparsity"
  — <https://arxiv.org/abs/2101.03961>.
- **`torchutils` / `auxloss`** — the `reg_loss` mechanism:
  <https://github.com/newsniper-org/torchutils>.
