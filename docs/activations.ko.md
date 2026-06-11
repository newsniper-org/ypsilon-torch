# 활성화 함수 (확장)

> English version: [activations.en.md](activations.en.md)

## 목차

1. [개요](#1-개요)
2. [수학적 정의](#2-수학적-정의)
3. [API 레퍼런스](#3-api-레퍼런스)
4. [사용 예제](#4-사용-예제)
5. [구현 노트](#5-구현-노트)
6. [참고문헌](#6-참고문헌)

---

## 1. 개요

`ypsilon_torch.blocks.activations`는 real 및 complex 텐서를 위한 활성화
함수 블록을 제공한다. 기존 `HGLU`/`ThASh`(real), `StableModReLU`/
`StableComplexCardioid`(complex) 외에 다음을 추가한다:

| 블록 | 도메인 | 핵심 아이디어 |
|------|--------|---------------|
| `ArSinh` | Real | log-압축 항등 대체 (heavy-tail 친화) |
| `ThAShGLU` | Real | ThASh 게이트 GLU, 출력 차원 절반 |
| `ComplexThASh` | Complex | 위상 보존 magnitude-ThASh ($|\cdot| < 1$) |
| `ComplexHGLU` | Complex | 위상 보존 magnitude-HGLU |
| `ZReLU` | Complex | 1사분면 통과 (Guberman, 2016) |
| `CReLU` | Complex | 실수부/허수부 독립 ReLU (Trabelsi et al., 2018) |

complex 블록 중 두 개는 real 블록과 짝을 이룬다. `ComplexThASh`는 real
`ThASh`의, `ComplexHGLU`는 real `HGLU`의 complex-도메인 대응물로, 동일한
스칼라 함수를 modulus $|z|$에 적용하고 위상 $\arg(z)$는 보존한다.

각 블록은 `nn.Module`이며, 동일한 이름의 functional 함수
(`ypsilon_torch.functional`)를 감싸는 얇은 래퍼이다.

---

## 2. 수학적 정의

### 2.1 ArSinh

$$
\operatorname{ArSinh}(x) = \operatorname{asinh}(x) = \log\!\left(x + \sqrt{1 + x^2}\right)
$$

홀함수이고 unbounded이며 log-압축적이다: 원점 근처에서는 항등함수처럼,
꼬리에서는 $\operatorname{sign}(x)\cdot\log(2|x|)$처럼 동작한다.

### 2.2 ThAShGLU

입력을 `dim` 축을 따라 $(a, b)$로 chunk 한 뒤:

$$
\operatorname{ThAShGLU}(x) = a \odot \operatorname{ThASh}(b),
\qquad
\operatorname{ThASh}(b) = \frac{b}{\sqrt{1 + b^2}}
$$

게이트 $\operatorname{ThASh}(b)$는 $(-1, 1)$로 bounded이며 sigmoid 게이트보다
더 완만하게 saturate 한다. `nn.GLU`와 마찬가지로 출력은 `dim`의 크기를
절반으로 줄인다 (해당 차원의 크기는 짝수여야 한다).

### 2.3 ComplexThASh

$$
f(z) = \frac{z}{|z|} \cdot \operatorname{ThASh}(|z|) = \frac{z}{\sqrt{1 + |z|^2}}
$$

modulus를 $(0, 1)$로 압축하면서 $\arg(z)$는 그대로 둔다. real `ThASh`의
complex-도메인 대응물이다. 파라미터가 없고 shape를 보존한다. $z = 0$에서의
특이점은 해석적으로 제거된다 (위 수식은 $|z|$로 나누지 않는다).

### 2.4 ComplexHGLU

$$
f(z) = \frac{z}{|z|} \cdot \operatorname{HGLU}_k(|z|),
\qquad
\operatorname{HGLU}_k(r) = \frac{r + \sqrt{k + r^2}}{2}
$$

modulus를 $\operatorname{HGLU}$ (치역 $(0, +\infty)$)를 통해 매핑하면서
$\arg(z)$는 보존한다. real `HGLU`의 complex-도메인 대응물이다.
$\epsilon$-smoothed magnitude $|z| \approx \sqrt{|z|^2 + \epsilon}$로 $z = 0$
지점을 well-defined 하게 유지한다.

### 2.5 ZReLU

$$
f(z) =
\begin{cases}
z & \text{if } \operatorname{Re}(z) > 0 \text{ and } \operatorname{Im}(z) > 0 \\
0 & \text{otherwise}
\end{cases}
$$

복소 평면의 (열린) 1사분면에 있을 때만 값을 통과시킨다. 파라미터가 없고
shape를 보존한다.

### 2.6 CReLU

$$
f(z) = \operatorname{ReLU}(\operatorname{Re}(z)) + i \cdot \operatorname{ReLU}(\operatorname{Im}(z))
$$

실수부와 허수부에 각각 독립적으로 ReLU를 적용한다. 파라미터가 없고
shape를 보존한다.

---

## 3. API 레퍼런스

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

인자가 없다. 입력 `(*)` → 출력 `(*)` (동일 shape).

#### `ThAShGLU`

```python
ThAShGLU(dim: int = -1)
```

| 파라미터 | 설명 |
|----------|------|
| `dim` | 게이트를 분할할 차원. 기본 `-1`. 해당 차원의 크기는 짝수여야 한다 |

입력 `(*, 2D, *)` → 출력 `(*, D, *)` (`dim` 절반).

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

| 파라미터 | 설명 |
|----------|------|
| `eps` | magnitude 평활화 epsilon |

real `ThASh`의 complex 대응물. 파라미터가 없고 shape를 보존한다.

#### `ComplexHGLU`

```python
ComplexHGLU(k: float, eps: float = 1e-12)
```

| 파라미터 | 설명 |
|----------|------|
| `k` | $\operatorname{HGLU}$의 양수 하이퍼파라미터 ($k > 0$ 필수) |
| `eps` | magnitude 평활화 epsilon. 기본 `1e-12` |

real `HGLU`의 complex 대응물. shape를 보존한다.

#### `ZReLU`

```python
ZReLU()
```

인자가 없다. shape를 보존한다.

#### `CReLU`

```python
CReLU()
```

인자가 없다. shape를 보존한다.

---

## 4. 사용 예제

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

functional API로도 동일하게 호출할 수 있다:

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

## 5. 구현 노트

- 각 `nn.Module` 블록은 동일한 이름의 `ypsilon_torch.functional` 함수를
  감싸는 얇은 래퍼이다. 상태(state)가 없으며 학습 가능한 파라미터를 갖지
  않는다.
- real/complex 쌍 구조: `ComplexThASh`↔`ThASh`, `ComplexHGLU`↔`HGLU`.
  complex 변형은 동일한 스칼라 함수를 modulus $|z|$에 적용하고 위상
  $\arg(z)$를 보존하므로, $f(z) = (z / |z|) \cdot g(|z|)$ 형태로 표현된다.
- `ComplexThASh`는 $z / \sqrt{1 + |z|^2}$로 닫힌 형태이므로 $z = 0$에서
  $|z|$로 나누지 않는다 — 특이점이 해석적으로 제거된다.
- `ComplexHGLU`는 $\epsilon$-smoothed magnitude $\sqrt{|z|^2 + \epsilon}$를
  사용해 $z = 0$ 근방의 $0/0$을 방지한다.
- functional 함수들은 real 입력도 허용한다: real 텐서가 들어오면
  `zrelu`/`crelu`는 일반 `relu`로, magnitude 기반 함수들은 $|x|$ 대신 $|x|$의
  real 정의로 자연스럽게 동작한다.

---

## 6. 참고문헌

1. N. Guberman, *On Complex Valued Convolutional Neural Networks* (zReLU), 2016. [arXiv:1602.09046](https://arxiv.org/abs/1602.09046)
2. C. Trabelsi et al., *Deep Complex Networks* (CReLU), ICLR 2018. [arXiv:1705.09792](https://arxiv.org/abs/1705.09792)

`ThASh` / `HGLU` 및 그 complex 대응물(`ComplexThASh` / `ComplexHGLU`)은
본 라이브러리 고유의 활성화이다.
