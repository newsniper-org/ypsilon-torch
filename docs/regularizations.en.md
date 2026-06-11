# Regularization Blocks

> 한국어 버전: [regularizations.ko.md](regularizations.ko.md)

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Definitions](#2-mathematical-definitions)
3. [API Reference](#3-api-reference)
4. [Usage Examples](#4-usage-examples)
5. [Hyperparameter Guide](#5-hyperparameter-guide)
6. [Implementation Notes](#6-implementation-notes)
7. [References](#7-references)

---

## 1. Overview

`ypsilon_torch.blocks.regularizations` provides stochastic regularization
blocks. All of them are **active only during training** and act as the
identity in evaluation (eval) mode.

| Block | Domain | Key Idea |
|-------|--------|----------|
| `GaussianNoise` | Real / Complex | additive Gaussian noise $y = x + \sigma\varepsilon$ |
| `ArsinhGaussianNoise` | Real | noise in arsinh space (heavy-tail-friendly) |
| `ComplexDropout` | Complex | dropout in the complex domain (details omitted) |

`ComplexDropout` is a dropout that applies a real-valued mask per complex
element, and is kept unchanged from its existing implementation. This document
focuses on the newly added `GaussianNoise` and `ArsinhGaussianNoise`.

---

## 2. Mathematical Definitions

### 2.1 GaussianNoise

$$
y = x + \sigma \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)
$$

Noise is added only during training; at evaluation time $y = x$ (identity).
Both real and complex tensors are supported. For complex inputs, independent
noise with standard deviation $\sigma$ is added to each of the real and
imaginary parts:

$$
y = x + \sigma \cdot (\varepsilon_{\mathrm{re}} + i\,\varepsilon_{\mathrm{im}}),
\qquad \varepsilon_{\mathrm{re}}, \varepsilon_{\mathrm{im}} \sim \mathcal{N}(0, 1)
$$

### 2.2 ArsinhGaussianNoise

$$
y = c \cdot \sinh\!\big( \operatorname{arsinh}(x / c) + \sigma \cdot \varepsilon \big),
\qquad \varepsilon \sim \mathcal{N}(0, 1)
$$

Because $\operatorname{arsinh}$ compresses large magnitudes logarithmically,
the same noise acts roughly **multiplicatively** on large-magnitude entries and
roughly **additively** on small ones. The result is a robust, heavy-tail-friendly
noise. Real inputs only, and active only during training. The scale $c$ sets the
magnitude scale at which the noise transitions from additive to multiplicative
behaviour.

---

## 3. API Reference

### Blocks (`nn.Module`)

```python
from ypsilon_torch.blocks.regularizations.real import (
    GaussianNoise,
    ArsinhGaussianNoise,
)
```

#### `GaussianNoise`

```python
GaussianNoise(sigma: float = 0.1)
```

| Parameter | Description |
|-----------|-------------|
| `sigma` | Standard deviation of the injected noise. Non-negative |

`forward(x)` branches on `self.training`: $x + \sigma\varepsilon$ during
training, $x$ during evaluation. Both real and complex tensors are supported.

#### `ArsinhGaussianNoise`

```python
ArsinhGaussianNoise(sigma: float = 0.1, c: float = 1.0)
```

| Parameter | Description |
|-----------|-------------|
| `sigma` | Noise standard deviation in arsinh space. Non-negative |
| `c` | arsinh scale. Magnitude at which noise transitions additive↔multiplicative. Positive |

`forward(x)` branches on `self.training`. Real inputs only.

### Functional (`functional`)

The blocks are thin wrappers over the functional implementations below. They
can be imported directly from `ypsilon_torch.functional`.

```python
gaussian_noise(x, sigma=0.1, training=True) -> Tensor
arsinh_gaussian_noise(x, sigma=0.1, c=1.0, training=True) -> Tensor
```

| Function | Parameters | Description |
|----------|------------|-------------|
| `gaussian_noise` | `x, sigma, training` | $x + \sigma\varepsilon$. real/complex. Identity if `training=False` |
| `arsinh_gaussian_noise` | `x, sigma, c, training` | noise in arsinh space. real only. Identity if `training=False` |

---

## 4. Usage Examples

```python
import torch
from ypsilon_torch.blocks.regularizations.real import (
    GaussianNoise, ArsinhGaussianNoise,
)

x = torch.randn(2, 128, 256)

# GaussianNoise — noise injected in training mode
gn = GaussianNoise(sigma=0.1)
gn.train()
y_train = gn(x)                                   # x != y_train
print(torch.allclose(x, y_train))                 # False

# Identity in evaluation mode
gn.eval()
y_eval = gn(x)                                     # x == y_eval
print(torch.allclose(x, y_eval))                  # True

# ArsinhGaussianNoise — heavy-tail-friendly noise
agn = ArsinhGaussianNoise(sigma=0.1, c=1.0)
agn.train()
y2_train = agn(x)                                  # x != y2_train
agn.eval()
y2_eval = agn(x)                                   # x == y2_eval
print(torch.allclose(x, y2_eval))                 # True

# GaussianNoise also supports complex tensors
z = torch.randn(2, 128, 256, dtype=torch.complex64)
gn.train()
w = gn(z)                                           # (2, 128, 256) complex64
```

---

## 5. Hyperparameter Guide

### `sigma` (GaussianNoise / ArsinhGaussianNoise)

| Value | Characteristics |
|-------|-----------------|
| $\sigma \to 0$ | No noise. Converges to identity |
| Small $\sigma$ (e.g. 0.05–0.1) | **Recommended starting point**. Light regularization |
| Large $\sigma$ | Strong perturbation. Excess can harm the training signal |

### `c` (ArsinhGaussianNoise)

- Large $c$: arsinh becomes nearly linear → converges to additive Gaussian noise.
- Small $c$: stronger non-linear compression → the multiplicative behaviour on
  large-magnitude entries becomes stronger.
- $c$ is interpreted as the magnitude scale at which the noise transitions from
  additive to multiplicative.

---

## 6. Implementation Notes

- Every block branches in `forward` based on `self.training`. In evaluation mode
  (`module.eval()`), no noise is added and the input is returned as-is.
- Unlike dropout, **no rescaling (e.g. a $1/(1-p)$ correction) is applied.** The
  expectation of additive noise equals the input ($\mathbb{E}[\varepsilon] = 0$),
  so no correction is needed.
- The complex case of `GaussianNoise` adds independent standard normal noise to
  the real and imaginary parts, then casts to the input dtype.
- The $c$ in `ArsinhGaussianNoise` sets the magnitude scale at which the noise
  transitions from additive to multiplicative behaviour. It short-circuits to
  the identity when `sigma == 0.0` or `training=False`.

---

## 7. References

1. Bishop, C. M. (1995). "Training with Noise is Equivalent to Tikhonov
   Regularization." *Neural Computation*, 7(1), 108–116.
2. Sietsma, J., & Dow, R. J. F. (1991). "Creating artificial neural networks
   that generalize." *Neural Networks*, 4(1), 67–79.
3. The arsinh (inverse hyperbolic sine) transform in general — a sign-preserving
   transform that compresses large magnitudes logarithmically.
