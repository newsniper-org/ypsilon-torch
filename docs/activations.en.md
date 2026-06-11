# Activation Functions (extensions)

> 한국어 버전: [activations.ko.md](activations.ko.md)

## Contents

1. [Overview](#1-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [API Reference](#3-api-reference)
4. [Usage](#4-usage)
5. [Implementation Notes](#5-implementation-notes)
6. [References](#6-references)

---

## 1. Overview

`ypsilon_torch.blocks.activations` provides activation-function blocks for
both real and complex tensors. In addition to the existing `HGLU` / `ThASh`
(real) and `StableModReLU` / `StableComplexCardioid` (complex), the following
blocks are added:

| Block | Domain | Key Idea |
|-------|--------|----------|
| `ArSinh` | Real | log-compressive identity replacement (heavy-tail friendly) |
| `ThAShGLU` | Real | ThASh-gated GLU, output dimension halved |
| `ComplexThASh` | Complex | phase-preserving magnitude-ThASh ($|\cdot| < 1$) |
| `ComplexHGLU` | Complex | phase-preserving magnitude-HGLU |
| `ZReLU` | Complex | first-quadrant pass-through (Guberman, 2016) |
| `CReLU` | Complex | independent ReLU on real / imaginary parts (Trabelsi et al., 2018) |

Two of the complex blocks are paired with real blocks. `ComplexThASh` is the
complex-domain analogue of the real `ThASh`, and `ComplexHGLU` of the real
`HGLU`: each applies the same scalar function to the modulus $|z|$ while
preserving the phase $\arg(z)$.

Every block is an `nn.Module` and a thin wrapper over the identically named
functional (`ypsilon_torch.functional`).

---

## 2. Mathematical Formulation

### 2.1 ArSinh

$$
\operatorname{ArSinh}(x) = \operatorname{asinh}(x) = \log\!\left(x + \sqrt{1 + x^2}\right)
$$

Odd, unbounded and log-compressive: it behaves like the identity near the
origin and like $\operatorname{sign}(x)\cdot\log(2|x|)$ in the tails.

### 2.2 ThAShGLU

After chunking the input into $(a, b)$ along `dim`:

$$
\operatorname{ThAShGLU}(x) = a \odot \operatorname{ThASh}(b),
\qquad
\operatorname{ThASh}(b) = \frac{b}{\sqrt{1 + b^2}}
$$

The gate $\operatorname{ThASh}(b)$ is bounded to $(-1, 1)$ and saturates more
gently than a sigmoid gate. As with `nn.GLU`, the output halves the size of
`dim` (which must be even).

### 2.3 ComplexThASh

$$
f(z) = \frac{z}{|z|} \cdot \operatorname{ThASh}(|z|) = \frac{z}{\sqrt{1 + |z|^2}}
$$

Squashes the modulus into $(0, 1)$ while leaving $\arg(z)$ untouched — the
complex-domain analogue of the real `ThASh`. Parameter-free and
shape-preserving. The singularity at $z = 0$ is removed analytically (the
formula above never divides by $|z|$).

### 2.4 ComplexHGLU

$$
f(z) = \frac{z}{|z|} \cdot \operatorname{HGLU}_k(|z|),
\qquad
\operatorname{HGLU}_k(r) = \frac{r + \sqrt{k + r^2}}{2}
$$

Maps the modulus through $\operatorname{HGLU}$ (range $(0, +\infty)$) while
preserving $\arg(z)$ — the complex-domain analogue of the real `HGLU`. An
$\epsilon$-smoothed magnitude $|z| \approx \sqrt{|z|^2 + \epsilon}$ keeps the
$z = 0$ point well-defined.

### 2.5 ZReLU

$$
f(z) =
\begin{cases}
z & \text{if } \operatorname{Re}(z) > 0 \text{ and } \operatorname{Im}(z) > 0 \\
0 & \text{otherwise}
\end{cases}
$$

Passes a value through only when it lies in the (open) first quadrant of the
complex plane. Parameter-free and shape-preserving.

### 2.6 CReLU

$$
f(z) = \operatorname{ReLU}(\operatorname{Re}(z)) + i \cdot \operatorname{ReLU}(\operatorname{Im}(z))
$$

Applies ReLU independently to the real and imaginary parts. Parameter-free
and shape-preserving.

---

## 3. API Reference

### Real

```python
from ypsilon_torch.blocks.activations.real import (
    ArSinh,
    ThAShGLU,
    HGLU,      # 기존
    ThASh,     # 기존
)
```

#### `ArSinh`

```python
ArSinh()
```

Takes no arguments. Input `(*)` → output `(*)` (same shape).

#### `ThAShGLU`

```python
ThAShGLU(dim: int = -1)
```

| Parameter | Description |
|-----------|-------------|
| `dim` | Dimension along which the gate is split. Default `-1`. Its size must be even |

Input `(*, 2D, *)` → output `(*, D, *)` (`dim` halved).

### Complex

```python
from ypsilon_torch.blocks.activations.complex import (
    ComplexThASh,
    ComplexHGLU,
    ZReLU,
    CReLU,
    StableModReLU,           # 기존
    StableComplexCardioid,   # 기존
)
```

#### `ComplexThASh`

```python
ComplexThASh(eps: float = 1e-12)
```

| Parameter | Description |
|-----------|-------------|
| `eps` | Magnitude-smoothing epsilon |

Complex analogue of the real `ThASh`. Parameter-free and shape-preserving.

#### `ComplexHGLU`

```python
ComplexHGLU(k: float, eps: float = 1e-12)
```

| Parameter | Description |
|-----------|-------------|
| `k` | Positive hyperparameter of $\operatorname{HGLU}$ ($k > 0$ required) |
| `eps` | Magnitude-smoothing epsilon. Default `1e-12` |

Complex analogue of the real `HGLU`. Shape-preserving.

#### `ZReLU`

```python
ZReLU()
```

Takes no arguments. Shape-preserving.

#### `CReLU`

```python
CReLU()
```

Takes no arguments. Shape-preserving.

---

## 4. Usage

```python
import torch
from ypsilon_torch.blocks.activations.real import ArSinh, ThAShGLU
from ypsilon_torch.blocks.activations.complex import (
    ComplexThASh, ComplexHGLU, ZReLU, CReLU,
)

# Real — ArSinh (shape 보존)
x = torch.randn(2, 128, 256)
act = ArSinh()
y = act(x)                                        # (2, 128, 256)

# Real — ThAShGLU (마지막 차원 절반)
x_glu = torch.randn(2, 128, 512)
glu = ThAShGLU(dim=-1)
y_glu = glu(x_glu)                                # (2, 128, 256)

# Complex — (B, seq_len, d_model) complex64
z = torch.randn(2, 128, 256, dtype=torch.complex64)

cthash = ComplexThASh()
w1 = cthash(z)                                    # (2, 128, 256) complex64

chglu = ComplexHGLU(k=1.0)
w2 = chglu(z)                                     # (2, 128, 256) complex64

w3 = ZReLU()(z)                                   # (2, 128, 256) complex64
w4 = CReLU()(z)                                   # (2, 128, 256) complex64
```

The functional API exposes the same operations:

```python
import torch
import ypsilon_torch.functional as F

x = torch.randn(2, 128, 256)
y = F.arsinh(x)                                   # (2, 128, 256)
y_glu = F.thash_glu(torch.randn(2, 128, 512))     # (2, 128, 256)

z = torch.randn(2, 128, 256, dtype=torch.complex64)
w1 = F.complex_thash(z)
w2 = F.complex_hglu(z, k=1.0)
w3 = F.zrelu(z)
w4 = F.crelu(z)
```

---

## 5. Implementation Notes

- Each `nn.Module` block is a thin wrapper over the identically named
  `ypsilon_torch.functional` function. The blocks are stateless and carry no
  learnable parameters.
- Real/complex pairing: `ComplexThASh`↔`ThASh`, `ComplexHGLU`↔`HGLU`. The
  complex variants apply the same scalar function to the modulus $|z|$ and
  preserve the phase $\arg(z)$, hence the form
  $f(z) = (z / |z|) \cdot g(|z|)$.
- `ComplexThASh` is closed-form as $z / \sqrt{1 + |z|^2}$, so it never divides
  by $|z|$ at $z = 0$ — the singularity is removed analytically.
- `ComplexHGLU` uses the $\epsilon$-smoothed magnitude
  $\sqrt{|z|^2 + \epsilon}$ to avoid the $0/0$ near $z = 0$.
- The functional functions also accept real input: for a real tensor
  `zrelu` / `crelu` reduce to an ordinary `relu`, and the magnitude-based
  functions fall back to the real definition of $|x|$ naturally.

---

## 6. References

1. N. Guberman, *On Complex Valued Convolutional Neural Networks* (zReLU), 2016. [arXiv:1602.09046](https://arxiv.org/abs/1602.09046)
2. C. Trabelsi et al., *Deep Complex Networks* (CReLU), ICLR 2018. [arXiv:1705.09792](https://arxiv.org/abs/1705.09792)

`ThASh` / `HGLU` and their complex counterparts (`ComplexThASh` /
`ComplexHGLU`) are activations original to this library.
