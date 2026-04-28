# N-dimensional Pooling Blocks

> 한국어 버전: [nd_pooling.ko.md](nd_pooling.ko.md)

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Definitions](#2-mathematical-definitions)
3. [Architecture & API Reference](#3-architecture--api-reference)
4. [Usage Examples](#4-usage-examples)
5. [Hyperparameter Guide](#5-hyperparameter-guide)
6. [Implementation Notes](#6-implementation-notes)

---

## 1. Overview

`ypsilon_torch.blocks.pooling` provides **unified 1D/2D/3D** robust pooling
blocks. They serve as drop-in replacements for `nn.AvgPool{1,2,3}d` or
`nn.MaxPool{1,2,3}d` with outlier-robust aggregation.

Each block comes in an **L_p specialisation** and a **general version**
accepting an arbitrary distance function, for both real and complex domains.

| Category | Real | Complex |
|----------|------|---------|
| arsinh average | `AsinhAvgPoolND` | `ComplexAsinhAvgPoolND` |
| arsinh RMS | `AsinhRMSPoolND` | `ComplexAsinhRMSPoolND` |
| Fréchet median (L_p) | `FrechetMedianLpPoolND` | `ComplexFrechetMedianLpPoolND` |
| Fréchet median (general) | `FrechetMedianPoolND` | `ComplexFrechetMedianPoolND` |
| Fréchet medoid (L_p) | `FrechetMedoidLpPoolND` | `ComplexFrechetMedoidLpPoolND` |
| Fréchet medoid (general) | `FrechetMedoidPoolND` | `ComplexFrechetMedoidPoolND` |
| Root-median-square (L_p) | `RootFrechetMedianLpSquarePoolND` | `ComplexRootFrechetMedianLpSquarePoolND` |
| Root-median-square (general) | `RootFrechetMedianSquarePoolND` | `ComplexRootFrechetMedianSquarePoolND` |
| Root-medoid-square (L_p) | `RootFrechetMedoidLpSquarePoolND` | `ComplexRootFrechetMedoidLpSquarePoolND` |
| Root-medoid-square (general) | `RootFrechetMedoidSquarePoolND` | `ComplexRootFrechetMedoidSquarePoolND` |

---

## 2. Mathematical Definitions

### Common notation

- Input $\mathbf{X} \in \mathbb{R}^{B \times C \times S_1 \times \cdots \times S_m}$
  (or $\mathbb{C}$).
- Output $\mathbf{Y} \in \mathbb{R}^{B \times C \times O_1 \times \cdots \times O_m}$.
- $\Omega(o)$: local window for output position $o$.

### Pooling operations

$$
\textbf{AsinhAvgPool}: \quad
\mathbf{Y}_{b,c,o} = \operatorname{SinhMeanArsinh}_{\alpha_c}\!\bigl(\{
  \mathbf{X}_{b,c,s} : s \in \Omega(o) \}\bigr)
$$

$$
\textbf{AsinhRMSPool}: \quad
\mathbf{Y}_{b,c,o} = \operatorname{SinhRMSArsinh}_{\alpha_c}\!\bigl(\{
  \mathbf{X}_{b,c,s} : s \in \Omega(o) \}\bigr)
$$

$$
\textbf{FréchetMedianPool}: \quad
\mathbf{Y}_{b,c,o} = \operatorname{FréchetMedian}_\rho\!\bigl(\{
  \mathbf{X}_{b,c,s} : s \in \Omega(o) \}\bigr)
$$

$$
\textbf{FréchetMedoidPool}: \quad
\mathbf{Y}_{b,c,o} = \operatorname{FréchetMedoid}_\rho\!\bigl(\{
  \mathbf{X}_{b,c,s} : s \in \Omega(o) \}\bigr)
$$

$$
\textbf{RootFréchetMedianSquarePool}: \quad
\mathbf{Y}_{b,c,o} = \operatorname{RootFréchetMedianSquare}_\rho\!\bigl(\{
  \mathbf{X}_{b,c,s} : s \in \Omega(o) \}\bigr)
$$

$$
\textbf{RootFréchetMedoidSquarePool}: \quad
\mathbf{Y}_{b,c,o} = \operatorname{RootFréchetMedoidSquare}_\rho\!\bigl(\{
  \mathbf{X}_{b,c,s} : s \in \Omega(o) \}\bigr)
$$

### Complex Asinh pooling — phase handling

Complex Asinh variants apply arsinh statistics to the **magnitude** and
determine the **phase** via circular mean:

$$
\theta_{\text{out}} = \operatorname{atan2}\!\Bigl(
  \sum_{s \in \Omega} \sin\theta_s,\;
  \sum_{s \in \Omega} \cos\theta_s
\Bigr)
$$

$$
\mathbf{Y}_{b,c,o} = r_{\text{pooled}} \cdot e^{i\theta_{\text{out}}}
$$

---

## 3. Architecture & API Reference

### Common parameters

Shared by all pooling blocks:

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel_size` | `int \| tuple[int, ...]` | Pooling window size |
| `stride` | `int \| tuple \| None` | Stride. `None` defaults to `kernel_size` |
| `padding` | `int \| tuple` | Zero-padding |

The number of spatial dimensions is inferred from `x.ndim - 2`.

### Real

```python
from ypsilon_torch.blocks.pooling.real import (
    # Asinh (learnable parameters)
    AsinhAvgPoolND,
    AsinhRMSPoolND,
    # L_p specialisations
    FrechetMedianLpPoolND,
    FrechetMedoidLpPoolND,
    RootFrechetMedianLpSquarePoolND,
    RootFrechetMedoidLpSquarePoolND,
    # General (dist_fn)
    FrechetMedianPoolND,
    FrechetMedoidPoolND,
    RootFrechetMedianSquarePoolND,
    RootFrechetMedoidSquarePoolND,
)
```

#### Asinh family

```python
AsinhAvgPoolND(channels, kernel_size, stride=None, padding=0,
               alpha_init=1.0, dtype_idx=64)
AsinhRMSPoolND(channels, kernel_size, stride=None, padding=0,
               alpha_init=1.0, dtype_idx=64)
```

| Extra parameter | Description |
|-----------------|-------------|
| `channels` | Number of input channels. Creates per-channel learnable $\alpha_c$ |
| `alpha_init` | Initial value for $\alpha$ |

#### L_p specialisations

```python
FrechetMedianLpPoolND(kernel_size, stride=None, padding=0,
                      p=1.0, max_iter=20, tol=1e-6)
FrechetMedoidLpPoolND(kernel_size, stride=None, padding=0)
RootFrechetMedianLpSquarePoolND(kernel_size, stride=None, padding=0,
                                p=1.0, max_iter=20, tol=1e-6)
RootFrechetMedoidLpSquarePoolND(kernel_size, stride=None, padding=0)
```

#### General

```python
FrechetMedianPoolND(dist_fn, kernel_size, stride=None, padding=0,
                    max_iter=20, tol=1e-6)
FrechetMedoidPoolND(dist_fn, kernel_size, stride=None, padding=0)
RootFrechetMedianSquarePoolND(dist_fn, kernel_size, stride=None, padding=0,
                              max_iter=20, tol=1e-6)
RootFrechetMedoidSquarePoolND(dist_fn, kernel_size, stride=None, padding=0)
```

| Extra parameter | Description |
|-----------------|-------------|
| `dist_fn` | `Callable[[Tensor], Tensor]`. Distance function $\rho$ |

### Complex

```python
from ypsilon_torch.blocks.pooling.complex import (
    ComplexAsinhAvgPoolND,      ComplexAsinhRMSPoolND,
    ComplexFrechetMedianLpPoolND, ComplexFrechetMedoidLpPoolND,
    ComplexRootFrechetMedianLpSquarePoolND,
    ComplexRootFrechetMedoidLpSquarePoolND,
    ComplexFrechetMedianPoolND, ComplexFrechetMedoidPoolND,
    ComplexRootFrechetMedianSquarePoolND,
    ComplexRootFrechetMedoidSquarePoolND,
)
```

Signatures mirror their real counterparts. Complex Asinh variants take a
`channels` parameter.

---

## 4. Usage Examples

```python
import torch
from ypsilon_torch.blocks.pooling.real import (
    AsinhAvgPoolND, FrechetMedianLpPoolND, FrechetMedianPoolND,
)
from ypsilon_torch.blocks.pooling.complex import (
    ComplexAsinhAvgPoolND, ComplexFrechetMedianPoolND,
)

# --- 1D (B, C, L) ---
x1d = torch.randn(4, 16, 128)
pool_1d = AsinhAvgPoolND(16, kernel_size=4, stride=4)
y1d = pool_1d(x1d)                                    # (4, 16, 32)

# --- 2D (B, C, H, W) ---
x2d = torch.randn(4, 32, 64, 64)
pool_2d = FrechetMedianLpPoolND(kernel_size=2, stride=2)
y2d = pool_2d(x2d)                                    # (4, 32, 32, 32)

# --- 2D with Huber distance ---
huber = lambda r: torch.where(r < 1.0, 0.5 * r**2, r - 0.5)
pool_huber = FrechetMedianPoolND(huber, kernel_size=2, stride=2)
y2d_h = pool_huber(x2d)                               # (4, 32, 32, 32)

# --- 3D (B, C, D, H, W) ---
x3d = torch.randn(2, 8, 16, 16, 16)
pool_3d = FrechetMedianLpPoolND(kernel_size=2, stride=2)
y3d = pool_3d(x3d)                                    # (2, 8, 8, 8, 8)

# --- Complex 2D ---
z2d = torch.randn(4, 32, 64, 64, dtype=torch.complex64)
cpool = ComplexAsinhAvgPoolND(32, kernel_size=2, stride=2)
w2d = cpool(z2d)                                       # (4, 32, 32, 32) complex64

cpool_gen = ComplexFrechetMedianPoolND(huber, kernel_size=2, stride=2)
w2d_h = cpool_gen(z2d)                                 # (4, 32, 32, 32) complex64
```

---

## 5. Hyperparameter Guide

### `alpha` (Asinh family)

- Large $\alpha$: arsinh becomes nearly linear → behaves like `nn.AvgPool`.
- Small $\alpha$: stronger non-linear compression → more robust to outliers.
- Per-channel learnable, so the network adapts automatically.

### `p` (L_p specialisations)

| $p$ | Pooling behaviour |
|-----|-------------------|
| 1.0 | Pure median pooling (most outlier-robust) |
| 2.0 | Equivalent to average pooling |
| 1.5 | Midpoint |

### `dist_fn` (General)

```python
# Predefined distance function examples
l1      = lambda r: r                                  # L1 (median)
l2_sq   = lambda r: r ** 2                             # L2 squared (mean)
huber   = lambda r: torch.where(r < 1, 0.5*r**2, r-0.5)
logcosh = lambda r: torch.log(torch.cosh(r))
```

---

## 6. Implementation Notes

### Window extraction

`NDPoolBase._extract_windows` chains `Tensor.unfold()` for each spatial
dimension:

```
(B, C, *spatial) → (B, C, *out_spatial, K)
```

$K = \prod_i k_i$ is the number of kernel elements. This handles 1D/2D/3D
with a single code path.

### Complex Asinh — magnitude + circular mean

To avoid complex arsinh branch-cut issues:

1. **Magnitude**: apply `sinh_mean_arsinh` or `sinh_rms_arsinh` to $|z|$.
2. **Phase**: determine via circular mean of window elements.
3. Reconstruct complex output with `torch.polar(mag, phase)`.

### Medoid — non-differentiable

All medoid variants return `.detach()`ed results.
