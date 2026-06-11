# Robust Location & Scale Estimators (extensions)

> 한국어 버전: [robust_location.ko.md](robust_location.ko.md)

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [API Reference](#3-api-reference)
4. [Usage Examples](#4-usage-examples)
5. [Hyperparameter Guide](#5-hyperparameter-guide)
6. [Implementation Notes](#6-implementation-notes)
7. [References](#7-references)

---

## 1. Overview

The `ypsilon_torch.functional.robust_location` module provides **classical
robust location and scale estimators** as a functional interface. It
complements the [`robust_stats`](robust_stats.en.md) module (Fréchet
median/medoid, arsinh-space statistics) with M-estimators and
order-statistic-based estimators.

- **Huber location**: a convex M-estimator that is quadratic for small
  residuals and linear for large ones, giving bounded influence.
- **Tukey biweight location**: a redescending M-estimator; residuals beyond
  the tuning constant receive **zero** influence, fully rejecting gross
  outliers.
- **Trimmed / Winsorized mean**: order-statistic estimators that drop or
  clamp a fraction of the most extreme samples.
- **Median absolute deviation (MAD)** and **biweight midvariance**: robust
  scale estimators.

It also provides **`HuberLayerNorm`** and **`TrimmedLayerNorm`**, which wrap
these estimators as normalization blocks.

All functions operate along a single `dim` and return a tensor with that
dimension reduced (`keepdim=False` by default). **Real-valued inputs only.**

---

## 2. Mathematical Formulation

### 2.1 Median absolute deviation (MAD)

$$
\operatorname{MAD}_s(x)
= s \cdot \operatorname{med}_j \bigl\lvert x_j - \operatorname{med}(x) \bigr\rvert
$$

The default scale constant $s = 1.4826$ makes the MAD a consistent estimator
of the standard deviation $\sigma$ for Gaussian data. $s = 1.0$ gives the raw
MAD.

### 2.2 Huber location

An M-estimator minimising the Huber loss $\rho_\delta$:

$$
\mu_H = \operatorname*{arg\,min}_\mu \sum_j \rho_\delta\!\left(\frac{x_j - \mu}{s}\right),
\qquad s = \operatorname{MAD}(x)
$$

Solved by IRLS, with weight

$$
w_j = \min\!\left(1,\; \frac{\delta}{\lvert r_j \rvert}\right),
\qquad r_j = \frac{x_j - \mu}{s} .
$$

$\delta$ is the tuning constant in MAD-scaled residual units; the default
$1.345$ gives ~95% efficiency under Gaussian noise.

### 2.3 Tukey biweight (bisquare) location

A redescending M-estimator: residuals with $\lvert r \rvert > c$ receive
**zero** weight, so gross outliers are fully rejected.

$$
w_j = \left(1 - (r_j/c)^2\right)^2 \cdot \mathbb{1}\bigl[\lvert r_j \rvert \le c\bigr],
\qquad r_j = \frac{x_j - \mu}{s} .
$$

$c$ is the tuning constant in MAD-scaled residual units; the default $4.685$
gives ~95% efficiency under Gaussian noise.

### 2.4 Trimmed / Winsorized mean

After sorting along `dim`, $k = \lfloor \tau \cdot n \rfloor$ samples are
handled at each tail.

$$
\operatorname{TrimmedMean}_\tau(x)
= \frac{1}{n - 2k} \sum_{j = k+1}^{n-k} x_{(j)}
$$

The trimmed mean **discards** $k$ samples from each tail and averages the
rest. The Winsorized mean instead **clamps** them to the nearest retained
order statistic:

$$
\operatorname{WinsorizedMean}_\tau(x)
= \frac{1}{n} \sum_{j=1}^{n}
  \operatorname{clamp}\bigl(x_{(j)},\, x_{(k+1)},\, x_{(n-k)}\bigr)
$$

with $\tau \in [0, 0.5)$.

### 2.5 Biweight midvariance

A robust estimator of variance:

$$
\zeta_{bi}^2 = \frac{n \sum_{\lvert u_j \rvert < 1} (x_j - M)^2 (1 - u_j^2)^4}
                    {\left(\sum_{\lvert u_j \rvert < 1} (1 - u_j^2)(1 - 5 u_j^2)\right)^2},
\qquad u_j = \frac{x_j - M}{c \cdot \operatorname{MAD}}
$$

where $M$ is the median. The return value is already the midvariance
(variance), so take `sqrt` for the biweight mid-standard-deviation. Default
$c = 9.0$.

### 2.6 Normalization blocks

Let $\mu$ be a robust location and $s = \operatorname{MAD}(x)$ a robust scale:

$$
\operatorname{HuberLayerNorm}(x)_j
    = \gamma_j \frac{x_j - \mu_H}{s + \epsilon} + \beta_j,
\qquad \mu_H = \operatorname{huber\_location}(x)
$$

$$
\operatorname{TrimmedLayerNorm}(x)_j
    = \gamma_j \frac{x_j - \mu_T}{s + \epsilon} + \beta_j,
\qquad \mu_T = \operatorname{TrimmedMean}_\tau(x)
$$

---

## 3. API Reference

All functions are importable from `ypsilon_torch.functional`.

### Location estimators (M-estimators)

```python
huber_location(x, dim, delta=1.345, max_iter=20, tol=1e-6, eps=1e-12, keepdim=False) -> Tensor
tukey_biweight_location(x, dim, c=4.685, max_iter=20, tol=1e-6, eps=1e-12, keepdim=False) -> Tensor
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Tensor` | Input (real only) |
| `dim` | `int` | Reduction dimension |
| `delta` | `float` | Huber tuning constant (MAD-scaled residual units). Default `1.345` |
| `c` | `float` | Biweight tuning constant (MAD-scaled residual units). Default `4.685` |
| `max_iter` | `int` | Maximum IRLS iterations |
| `tol` | `float` | Convergence tolerance |
| `eps` | `float` | Numerical floor |
| `keepdim` | `bool` | Whether to keep the reduced dimension |

### Location estimators (order statistics)

```python
trimmed_mean(x, dim, trim=0.1, keepdim=False) -> Tensor
winsorized_mean(x, dim, trim=0.1, keepdim=False) -> Tensor
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Tensor` | Input (real only) |
| `dim` | `int` | Reduction dimension |
| `trim` | `float` | Fraction trimmed/Winsorized at *each* tail, in `[0, 0.5)`. Default `0.1` |
| `keepdim` | `bool` | Whether to keep the reduced dimension |

### Scale estimators

```python
median_abs_deviation(x, dim, scale=1.4826, keepdim=False) -> Tensor
biweight_midvariance(x, dim, c=9.0, eps=1e-12, keepdim=False) -> Tensor
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `Tensor` | Input (real only) |
| `dim` | `int` | Reduction dimension |
| `scale` | `float` | MAD consistency constant. Default `1.4826` (Gaussian-consistent). Use `1.0` for raw MAD |
| `c` | `float` | Biweight midvariance tuning constant. Default `9.0` |
| `eps` | `float` | Numerical floor |
| `keepdim` | `bool` | Whether to keep the reduced dimension |

### Normalization blocks

```python
from ypsilon_torch.blocks.normalizations.real import HuberLayerNorm, TrimmedLayerNorm

HuberLayerNorm(normalized_shape, delta=1.345, eps=1e-5, dim=-1,
               affine=True, dtype_idx=64, max_iter=10)
TrimmedLayerNorm(normalized_shape, trim=0.1, eps=1e-5, dim=-1,
                 affine=True, dtype_idx=64)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `normalized_shape` | `int` | Size of the dimension to normalize |
| `delta` | `float` | (Huber) Huber tuning constant. Default `1.345` |
| `trim` | `float` | (Trimmed) fraction trimmed from each tail, in `[0, 0.5)`. Default `0.1` |
| `eps` | `float` | Stability epsilon added to the scale. Default `1e-5` |
| `dim` | `int` | Dimension to normalize over. Default `-1` |
| `affine` | `bool` | If True, apply learnable `gamma`/`beta`. Default True |
| `dtype_idx` | `FPDTypeIdx` | Floating-point precision index. Default `64` (resolved to a dtype via `get_float_dtype`) |
| `max_iter` | `int` | (Huber) Maximum IRLS iterations. Default `10` |

---

## 4. Usage Examples

```python
import torch
from ypsilon_torch.functional import (
    median_abs_deviation,
    huber_location,
    tukey_biweight_location,
    trimmed_mean,
    winsorized_mean,
    biweight_midvariance,
)

x = torch.randn(8, 64)  # (batch, features)

# robust scale (MAD, Gaussian-consistent)
s = median_abs_deviation(x, dim=-1)                 # shape: (8,)

# Huber location (bounded influence)
mu_h = huber_location(x, dim=-1, delta=1.345)       # shape: (8,)

# Tukey biweight location (redescending)
mu_t = tukey_biweight_location(x, dim=-1, c=4.685)  # shape: (8,)

# trimmed / winsorized mean
mu_tr = trimmed_mean(x, dim=-1, trim=0.1)           # shape: (8,)
mu_wn = winsorized_mean(x, dim=-1, trim=0.1)        # shape: (8,)

# biweight midvariance (robust variance)
var = biweight_midvariance(x, dim=-1, c=9.0)        # shape: (8,)
```

Normalization blocks:

```python
import torch
from ypsilon_torch.blocks.normalizations.real import HuberLayerNorm, TrimmedLayerNorm

x = torch.randn(4, 16, 64, dtype=torch.float64)  # (batch, seq, features)

huber_ln = HuberLayerNorm(64)
trim_ln = TrimmedLayerNorm(64, trim=0.1)

y_h = huber_ln(x)   # shape: (4, 16, 64)
y_t = trim_ln(x)    # shape: (4, 16, 64)
```

---

## 5. Hyperparameter Guide

| Estimator | Constant | Default | Meaning |
|-----------|----------|---------|---------|
| `huber_location` | `delta` | `1.345` | ~95% Gaussian efficiency. Smaller is more robust (influence clipped sooner); larger approaches the mean |
| `tukey_biweight_location` | `c` | `4.685` | ~95% Gaussian efficiency. Residuals with $\lvert r \rvert > c$ are fully rejected. Smaller rejects outliers more aggressively |
| `trimmed_mean` / `winsorized_mean` | `trim` | `0.1` | Fraction handled at each tail. `0` is the plain mean; near `0.5` approaches the median |
| `median_abs_deviation` | `scale` | `1.4826` | Gaussian consistency constant. Use `1.0` for raw MAD |
| `biweight_midvariance` | `c` | `9.0` | Midvariance tuning constant (conventional value) |

Selection guide:

- Want **bounded but non-zero influence** with convexity (guaranteed
  convergence)? → Huber.
- Want **full rejection of gross outliers** (redescending)? → Tukey biweight.
  Note that it is non-convex and can be initialisation-sensitive (mitigated by
  the MAD/median initialisation).
- Want a **simple, fast, breakdown-bounded estimate**? → trimmed/Winsorized
  mean (a single sort).

---

## 6. Implementation Notes

### IRLS (Huber / Tukey biweight)

Both M-estimators share the same IRLS structure:

1. Initialise $\mu^{(0)} = \operatorname{median}(x)$ and
   $s = \operatorname{MAD}(x)$ (both clamped from below by `eps`).
2. Compute residuals $r = (x - \mu)/s$, then the estimator-specific weights
   $w_j$.
3. $\mu^{(k+1)} = \dfrac{\sum_j w_j x_j}{\sum_j w_j}$.
4. Stop on convergence (max of $\lvert \mu^{(k+1)} - \mu^{(k)} \rvert <$
   `tol`) or when `max_iter` is reached.

The IRLS loop runs inside `torch.no_grad()`. The returned location $\mu$
therefore behaves like a detached constant, and gradients flow only through
the $(x - \mu)/s$ path of the normalisation expression. This is the **same
convention** as `frechet_median_lp` in [`robust_stats`](robust_stats.en.md)
(the IRLS iterations do not propagate gradients; the location is treated as a
detached constant).

### Order statistics (trimmed / Winsorized)

After sorting with `torch.sort`, the trimmed mean uses `narrow` to drop $k$
samples from each tail and averages the rest. The Winsorized mean clamps to
the boundary order statistics via `torch.minimum`/`torch.maximum`. Unlike
IRLS, this is a closed-form single sort and is fully differentiable through
the retained samples.

### Gradient convention of the normalization blocks

`HuberLayerNorm` uses `huber_location` for the location and
`median_abs_deviation` for the scale. `TrimmedLayerNorm` uses `trimmed_mean`
for the location. Both blocks normalise as
$(x - \mu)/(s + \epsilon) \cdot \gamma + \beta$.

The Huber location is solved by IRLS (`no_grad`), so it behaves as a detached
constant and gradients flow only through the $(x - \mu)/s$ path — the same
convention as `frechet_median_lp` in robust_stats. The trimmed location is
differentiable through the retained samples. In both blocks the MAD scale also
passes through median operations and so behaves effectively as a detached
constant.

`dtype_idx` (`FPDTypeIdx`, default `64`) selects the floating-point precision
of the learnable parameters `gamma`/`beta` via `get_float_dtype`.

These blocks sit between the mean-based `AsinhMeanLayerNorm` and the
IRLS-based `RobustLayerNorm`, thanks to the bounded influence of the Huber
location.

---

## 7. References

1. Huber, P. J. (1964). "Robust Estimation of a Location Parameter."
   *Annals of Mathematical Statistics*, 35(1), 73–101.
2. Tukey, J. W. — biweight (bisquare) redescending M-estimator.
   Beaton, A. E., & Tukey, J. W. (1974). "The Fitting of Power Series,
   Meaning Polynomials, Illustrated on Band-Spectroscopic Data."
   *Technometrics*, 16(2), 147–185.
3. Hampel, F. R. (1974). "The Influence Curve and Its Role in Robust
   Estimation." (MAD as a robust scale estimator.)
4. [Robust gradient estimation](https://opt-ml.org/papers/2025/paper74.pdf)
