# Robust Statistics (Functional Utilities)

> 한국어 버전: [robust_stats.ko.md](robust_stats.ko.md)

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Definitions](#2-mathematical-definitions)
3. [API Reference](#3-api-reference)
4. [Usage Examples](#4-usage-examples)
5. [Implementation Notes](#5-implementation-notes)
6. [References](#6-references)

---

## 1. Overview

The `ypsilon_torch.functional.robust_stats` module provides **Fréchet
median/medoid** and **arsinh-space statistics** as a functional interface.

- **Fréchet median**: continuous location estimator that minimises total
  distance under an arbitrary distance function $\rho$.
- **Fréchet medoid**: discrete location estimator that picks the sample
  element minimising total distance.
- **arsinh-space statistics**: robust statistics computed in the
  $\operatorname{arsinh}$-transformed space, then mapped back via $\sinh$.

All functions support **both real and complex tensors** (except the arsinh
family, which is real-only).

---

## 2. Mathematical Definitions

### 2.1 Fréchet median & medoid

For a distance function $\rho : [0, +\infty) \to [0, +\infty)$,

$$
\operatorname{FréchetMedian}_\rho(X)
= \arg\min_{y \in M} \sum_{x \in X} \rho\!\bigl(\lvert x - y \rvert\bigr)
$$

$$
\operatorname{FréchetMedoid}_\rho(X)
= \arg\min_{y \in X} \sum_{x \in X} \rho\!\bigl(\lvert x - y \rvert\bigr)
$$

**L_p special cases** ($\rho(r) = r^p$):

| $p$ | Fréchet median | Notes |
|-----|----------------|-------|
| 1   | ordinary median | uses `torch.median` directly |
| 2   | arithmetic mean | uses `torch.mean` directly |
| $(1, 2)$ | IRLS iterative solver | |

For complex tensors the distance is the complex modulus
$\lvert z_i - z_j \rvert$, solved via the Weiszfeld algorithm for the
geometric median in the complex plane.

### 2.2 arsinh-space statistics

Transform pair: $f_c(x) = \operatorname{arsinh}(x/c)$, $g_c(z) = c \sinh(z)$.

$$
\operatorname{SinhMeanArsinh}_c(X) = g_c\!\bigl(\mathbb{E}[f_c(X)]\bigr)
$$

$$
\operatorname{SinhRMSArsinh}_c(X) = g_c\!\bigl(\operatorname{RMS}[f_c(X)]\bigr)
$$

$$
\operatorname{SinhFréchetMedianArsinh}_{c,\rho}(X)
= g_c\!\bigl(\operatorname{FréchetMedian}_\rho(f_c(X))\bigr)
$$

$$
\operatorname{SinhFréchetMedoidArsinh}_{c,\rho}(X)
= g_c\!\bigl(\operatorname{FréchetMedoid}_\rho(f_c(X))\bigr)
$$

### 2.3 Root-Fréchet-\*-Square

$$
\operatorname{RootFréchetMedianSquare}_\rho(X)
= \sqrt{\operatorname{FréchetMedian}_\rho(X^2)}
$$

$$
\operatorname{RootFréchetMedoidSquare}_\rho(X)
= \sqrt{\operatorname{FréchetMedoid}_\rho(X^2)}
$$

---

## 3. API Reference

All functions are importable from `ypsilon_torch.functional`.

### General (arbitrary distance function)

```python
frechet_median(x, dim, dist_fn, max_iter=20, tol=1e-6) -> Tensor
frechet_medoid(x, dim, dist_fn) -> Tensor
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Tensor` | Input (real or complex) |
| `dim` | `int` | Reduction dimension |
| `dist_fn` | `Callable[[Tensor], Tensor]` | Distance function $\rho$. Takes non-negative residual magnitudes, returns distances. Must be differentiable (median only) |
| `max_iter` | `int` | Maximum IRLS iterations |
| `tol` | `float` | Convergence tolerance |

The return value of `frechet_medoid` is `.detach()`ed (no gradient propagation).

### L_p specialisation

```python
frechet_median_lp(x, dim, p=1.0, max_iter=20, tol=1e-6) -> Tensor
frechet_medoid_lp(x, dim) -> Tensor
```

### arsinh compositions

```python
sinh_mean_arsinh(x, c, dim) -> Tensor
sinh_rms_arsinh(x, c, dim) -> Tensor
sinh_frechet_median_arsinh(x, c, dim, dist_fn, ...) -> Tensor
sinh_frechet_median_lp_arsinh(x, c, dim, p=1.0, ...) -> Tensor
sinh_frechet_medoid_arsinh(x, c, dim, dist_fn) -> Tensor
sinh_frechet_medoid_lp_arsinh(x, c, dim) -> Tensor
```

### Root-\*-Square compositions

```python
root_frechet_median_square(x, dim, dist_fn, ...) -> Tensor
root_frechet_median_lp_square(x, dim, p=1.0, ...) -> Tensor
root_frechet_medoid_square(x, dim, dist_fn) -> Tensor
root_frechet_medoid_lp_square(x, dim) -> Tensor
```

---

## 4. Usage Examples

```python
import torch
from ypsilon_torch.functional import (
    frechet_median_lp, frechet_median,
    sinh_mean_arsinh,
)

x = torch.randn(8, 64)  # (batch, features)

# L1 median (p=1)
med = frechet_median_lp(x, dim=-1, p=1.0)          # shape: (8,)

# L_1.5 median
med_15 = frechet_median_lp(x, dim=-1, p=1.5)       # shape: (8,)

# Huber-distance median
huber = lambda r: torch.where(r < 1.0, 0.5 * r**2, r - 0.5)
med_huber = frechet_median(x, dim=-1, dist_fn=huber)  # shape: (8,)

# arsinh-space mean
avg = sinh_mean_arsinh(x, c=1.0, dim=-1)           # shape: (8,)

# Complex geometric median
z = torch.randn(8, 64, dtype=torch.complex64)
gmed = frechet_median_lp(z, dim=-1)                 # shape: (8,), complex
```

---

## 5. Implementation Notes

### IRLS (Iteratively Reweighted Least Squares)

L_p median ($1 < p < 2$) solver:

1. Initialise $\mu^{(0)} = \operatorname{median}(x)$.
2. $w_j^{(k)} = \lvert x_j - \mu^{(k)} \rvert^{p-2}$, $\mu^{(k+1)} = \frac{\sum w_j x_j}{\sum w_j}$.
3. Stop on convergence or `max_iter` reached.

The IRLS loop runs inside `torch.no_grad()` to save memory.

### General distance — autodiff-based weight computation

IRLS weights for an arbitrary $\rho$ are:

$$
w_j = \frac{\rho'(r_j)}{r_j}, \qquad r_j = \lvert x_j - \mu \rvert
$$

$\rho'$ is computed automatically via `torch.autograd.grad`
(`torch.enable_grad()` is used locally inside the `no_grad` block).

### Weiszfeld (complex)

Geometric median solver in the complex plane. Same structure as real IRLS
but uses $w_j = 1 / \lvert z_j - \mu \rvert$ (L1 case) or autodiff weights
(general case).

### Medoid — non-differentiable

`frechet_medoid` performs a discrete `argmin`, so the result is
`.detach()`ed.

---

## 6. References

1. Weiszfeld, E. (1937). "Sur le point pour lequel la somme des distances de n
   points donnés est minimum."
2. [Robust gradient estimation](https://opt-ml.org/papers/2025/paper74.pdf)
