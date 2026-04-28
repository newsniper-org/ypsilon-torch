# Normalization Blocks

> 한국어 버전: [normalizations.ko.md](normalizations.ko.md)

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Definitions](#2-mathematical-definitions)
3. [Architecture & API Reference](#3-architecture--api-reference)
4. [Usage Examples](#4-usage-examples)
5. [Hyperparameter Guide](#5-hyperparameter-guide)
6. [Implementation Notes](#6-implementation-notes)
7. [References](#7-references)

---

## 1. Overview

`ypsilon_torch.blocks.normalizations` provides outlier-robust normalization
blocks for both real and complex tensors. In addition to the existing
`ComplexLayerNorm` / `ComplexRMSNorm`, the following blocks are added:

| Block | Domain | Key Idea |
|-------|--------|----------|
| `RobustLayerNorm` | Real | L_φ location estimator + MAD scale |
| `AsinhMeanLayerNorm` | Real | normalisation in arsinh-transformed space |
| `ComplexRobustLayerNorm` | Complex | Weiszfeld location estimator + MAD scale |
| `ComplexAsinhMeanLayerNorm` | Complex | magnitude-based arsinh normalisation |

---

## 2. Mathematical Definitions

### 2.1 RobustLayerNorm

$1 < \phi < 2$ (recommended: $1.2 \lesssim \phi < 2$).

$$
\mu_\phi = \arg\min_{y} \sum_{j=1}^{d} {\left| x_j - y \right|}^{\phi}
$$

$$
s_\phi = \operatorname{med}_j \left| x_j - \mu_\phi \right|
$$

$$
\operatorname{RobustLayerNorm}_\phi(x)_j
= \gamma_j \frac{x_j - \mu_\phi}{s_\phi + \epsilon} + \beta_j
$$

### 2.2 AsinhMeanLayerNorm

$$
z_j = \operatorname{arsinh}(x_j / c)
$$

$$
\operatorname{AsinhMeanLayerNorm}(X)_j
= \gamma_j \frac{z_j - \mathbb{E}(Z)}{\sqrt{\mathbb{V}(Z) + \epsilon}} + \beta_j
$$

$c$ is a learnable scalar parameter.

### 2.3 ComplexRobustLayerNorm

The location $\mu_\phi$ is computed via the Weiszfeld algorithm in the
complex plane. The scale $s_\phi$ is the MAD of the magnitudes (real).

$$
\operatorname{ComplexRobustLayerNorm}_\phi(z)_j
= \gamma_j \frac{z_j - \mu_\phi}{s_\phi + \epsilon} + \beta_j
$$

$\gamma, \beta$ are complex parameters. Phase is naturally preserved.

### 2.4 ComplexAsinhMeanLayerNorm

Magnitude-based approach (avoids complex arsinh branch cuts):

$$
m_j = |z_j|, \quad u_j = \operatorname{arsinh}(m_j / c)
$$

$$
\hat{u}_j = \frac{u_j - \mathbb{E}(U)}{\sqrt{\mathbb{V}(U) + \epsilon}}
$$

$$
\operatorname{out}_j = \gamma_j \cdot z_j \cdot \frac{\hat{u}_j}{m_j + \epsilon} + \beta_j
$$

$\gamma, \beta$ are complex parameters.

---

## 3. Architecture & API Reference

### Real

```python
from ypsilon_torch.blocks.normalizations.real import (
    RobustLayerNorm,
    AsinhMeanLayerNorm,
)
```

#### `RobustLayerNorm`

```python
RobustLayerNorm(
    normalized_shape: int,
    phi: float = 1.5,
    eps: float = 1e-5,
    dim: int = -1,
    affine: bool = True,
    dtype_idx: FPDTypeIdx = 64,
    max_iter: int = 10,
)
```

| Parameter | Description |
|-----------|-------------|
| `normalized_shape` | Size of the normalisation dimension |
| `phi` | L_φ location estimator exponent. Range $(1, 2)$ |
| `dim` | Normalisation dimension. Default `-1` |
| `affine` | If `True`, apply learnable `gamma` and `beta` |
| `max_iter` | Maximum IRLS iterations |

#### `AsinhMeanLayerNorm`

```python
AsinhMeanLayerNorm(
    normalized_shape: int,
    c_init: float = 1.0,
    eps: float = 1e-5,
    dim: int = -1,
    affine: bool = True,
    dtype_idx: FPDTypeIdx = 64,
)
```

| Parameter | Description |
|-----------|-------------|
| `c_init` | Initial value for the learnable arsinh scale `c` |

### Complex

```python
from ypsilon_torch.blocks.normalizations.complex import (
    ComplexRobustLayerNorm,
    ComplexAsinhMeanLayerNorm,
    ComplexLayerNorm,        # existing
    ComplexRMSNorm,          # refactored
)
```

#### `ComplexRobustLayerNorm`

Same signature as `RobustLayerNorm`. `gamma` and `beta` are complex-valued.

#### `ComplexAsinhMeanLayerNorm`

Same signature as `AsinhMeanLayerNorm`. `gamma` and `beta` are complex-valued,
`c` is a real scalar.

#### `ComplexRMSNorm` (refactored)

```python
ComplexRMSNorm(
    normalized_shape: int,
    eps: float = 1e-6,
    dim: int = -1,
    affine: bool = False,
    dtype_idx: FPDTypeIdx = 64,
)
```

Changes from prior version:
- `d_model` → `normalized_shape` renamed
- `dim` parameter added (default `-1`, previously hard-coded)
- `affine` flag added (default `False`)
- `reset_parameters()` method added

---

## 4. Usage Examples

```python
import torch
from ypsilon_torch.blocks.normalizations.real import (
    RobustLayerNorm, AsinhMeanLayerNorm,
)
from ypsilon_torch.blocks.normalizations.complex import (
    ComplexRobustLayerNorm, ComplexAsinhMeanLayerNorm,
)

# Real — Transformer style (B, seq_len, d_model)
x = torch.randn(2, 128, 256)
rln = RobustLayerNorm(256, phi=1.5)
y = rln(x)                                        # (2, 128, 256)

aln = AsinhMeanLayerNorm(256, c_init=1.0)
y2 = aln(x)                                       # (2, 128, 256)

# Real — CNN style (B, C, H, W), channel-dim normalisation
x_cnn = torch.randn(2, 64, 16, 16)
rln_ch = RobustLayerNorm(64, dim=1)
y3 = rln_ch(x_cnn)                                # (2, 64, 16, 16)

# Complex — (B, seq_len, d_model) complex64
z = torch.randn(2, 128, 256, dtype=torch.complex64)
crln = ComplexRobustLayerNorm(256)
w = crln(z)                                        # (2, 128, 256) complex64

caln = ComplexAsinhMeanLayerNorm(256)
w2 = caln(z)                                       # (2, 128, 256) complex64
```

---

## 5. Hyperparameter Guide

### `phi` (RobustLayerNorm / ComplexRobustLayerNorm)

| Value | Characteristics |
|-------|-----------------|
| $\phi \to 1^+$ | Close to ordinary median. Most robust to extreme outliers |
| $\phi = 1.5$ | **Recommended default**. Balances robustness and efficiency |
| $\phi \to 2^-$ | Close to mean. Increased outlier sensitivity |

Reference 1 recommends $1.2 \lesssim \phi < 2$.

### `c` (AsinhMeanLayerNorm / ComplexAsinhMeanLayerNorm)

- Large $c$: arsinh becomes nearly linear → converges to standard LayerNorm.
- Small $c$: stronger non-linear compression → more robust to outliers.
- Learnable, so only the initial value (`c_init`) needs to be set.

---

## 6. Implementation Notes

- IRLS / Weiszfeld loops run inside `torch.no_grad()` to save memory on
  intermediate computations. Only the final normalisation step propagates
  gradients.
- The `dim` parameter allows free choice of normalisation axis (`-1` = last
  dimension, `1` = channel dimension, etc.).
- `gamma` / `beta` broadcast shapes are built dynamically based on `dim`.

---

## 7. References

1. [Robust gradient estimation](https://opt-ml.org/papers/2025/paper74.pdf)
