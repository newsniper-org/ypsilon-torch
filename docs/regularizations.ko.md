# Regularization 블록

> English version: [regularizations.en.md](regularizations.en.md)

## 목차

1. [개요](#1-개요)
2. [수학적 정의](#2-수학적-정의)
3. [API 레퍼런스](#3-api-레퍼런스)
4. [사용 예제](#4-사용-예제)
5. [하이퍼파라미터 가이드](#5-하이퍼파라미터-가이드)
6. [구현 노트](#6-구현-노트)
7. [참고문헌](#7-참고문헌)

---

## 1. 개요

`ypsilon_torch.blocks.regularizations`는 확률적(stochastic) 정규화 블록을
제공한다. 모두 **학습 시에만** 동작하며 평가(eval) 모드에서는 identity로
동작한다.

| 블록 | 도메인 | 핵심 아이디어 |
|------|--------|---------------|
| `GaussianNoise` | Real / Complex | 가법 Gaussian 노이즈 $y = x + \sigma\varepsilon$ |
| `ArsinhGaussianNoise` | Real | arsinh 공간 노이즈 (heavy-tail 친화) |
| `ComplexDropout` | Complex | complex 도메인 dropout (자세한 내용 생략) |

`ComplexDropout`은 complex 원소 단위로 실수 mask를 적용하는 dropout으로,
기존 구현을 그대로 유지한다. 이 문서는 새로 추가된 `GaussianNoise`와
`ArsinhGaussianNoise`를 중심으로 설명한다.

---

## 2. 수학적 정의

### 2.1 GaussianNoise

$$
y = x + \sigma \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)
$$

학습 시에만 노이즈를 더하고, 평가 시에는 $y = x$ (identity)이다. Real 및
complex 텐서를 모두 지원하며, complex의 경우 실수부와 허수부 각각에 표준편차
$\sigma$의 독립 노이즈를 더한다:

$$
y = x + \sigma \cdot (\varepsilon_{\mathrm{re}} + i\,\varepsilon_{\mathrm{im}}),
\qquad \varepsilon_{\mathrm{re}}, \varepsilon_{\mathrm{im}} \sim \mathcal{N}(0, 1)
$$

### 2.2 ArsinhGaussianNoise

$$
y = c \cdot \sinh\!\big( \operatorname{arsinh}(x / c) + \sigma \cdot \varepsilon \big),
\qquad \varepsilon \sim \mathcal{N}(0, 1)
$$

$\operatorname{arsinh}$가 큰 크기를 로그적으로 압축하므로, 동일한 노이즈가
큰 크기 항에는 사실상 **곱셈적(multiplicative)** 으로, 작은 값에는 거의
**가법적(additive)** 으로 작용한다. 그 결과 heavy-tail 신호에 친화적인 robust
노이즈가 된다. Real 입력만 지원하며, 학습 시에만 동작한다. 스케일 $c$는 노이즈가
가법적에서 곱셈적으로 전이되는 크기 스케일을 정한다.

---

## 3. API 레퍼런스

### 블록 (`nn.Module`)

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

| 파라미터 | 설명 |
|----------|------|
| `sigma` | 주입 노이즈의 표준편차. non-negative |

`forward(x)`는 `self.training`에 따라 분기한다: 학습 시 $x + \sigma\varepsilon$,
평가 시 $x$. Real 및 complex 텐서를 모두 지원한다.

#### `ArsinhGaussianNoise`

```python
ArsinhGaussianNoise(sigma: float = 0.1, c: float = 1.0)
```

| 파라미터 | 설명 |
|----------|------|
| `sigma` | arsinh 공간에서의 노이즈 표준편차. non-negative |
| `c` | arsinh 스케일. 노이즈가 가법↔곱셈으로 전이되는 크기. positive |

`forward(x)`는 `self.training`에 따라 분기한다. Real 입력만 지원한다.

### 함수형 (`functional`)

블록은 아래 함수형 구현 위에 얇게 감싼 wrapper이다. `ypsilon_torch.functional`
에서 직접 import 할 수 있다.

```python
gaussian_noise(x, sigma=0.1, training=True) -> Tensor
arsinh_gaussian_noise(x, sigma=0.1, c=1.0, training=True) -> Tensor
```

| 함수 | 파라미터 | 설명 |
|------|----------|------|
| `gaussian_noise` | `x, sigma, training` | $x + \sigma\varepsilon$. real/complex. `training=False`이면 identity |
| `arsinh_gaussian_noise` | `x, sigma, c, training` | arsinh 공간 노이즈. real 전용. `training=False`이면 identity |

---

## 4. 사용 예제

```python
import torch
from ypsilon_torch.blocks.regularizations.real import (
    GaussianNoise, ArsinhGaussianNoise,
)

x = torch.randn(2, 128, 256)

# GaussianNoise — 학습 모드에서는 노이즈 주입
gn = GaussianNoise(sigma=0.1)
gn.train()
y_train = gn(x)                                   # x != y_train
print(torch.allclose(x, y_train))                 # False

# 평가 모드에서는 identity
gn.eval()
y_eval = gn(x)                                     # x == y_eval
print(torch.allclose(x, y_eval))                  # True

# ArsinhGaussianNoise — heavy-tail 친화 노이즈
agn = ArsinhGaussianNoise(sigma=0.1, c=1.0)
agn.train()
y2_train = agn(x)                                  # x != y2_train
agn.eval()
y2_eval = agn(x)                                   # x == y2_eval
print(torch.allclose(x, y2_eval))                 # True

# GaussianNoise는 complex 텐서도 지원
z = torch.randn(2, 128, 256, dtype=torch.complex64)
gn.train()
w = gn(z)                                           # (2, 128, 256) complex64
```

---

## 5. 하이퍼파라미터 가이드

### `sigma` (GaussianNoise / ArsinhGaussianNoise)

| 값 | 특성 |
|----|------|
| $\sigma \to 0$ | 노이즈 없음. identity에 수렴 |
| 작은 $\sigma$ (예: 0.05–0.1) | **권장 시작점**. 가벼운 정규화 |
| 큰 $\sigma$ | 강한 perturbation. 과도하면 학습 신호를 해칠 수 있음 |

### `c` (ArsinhGaussianNoise)

- 큰 $c$: arsinh가 거의 선형 → 가법 Gaussian 노이즈에 수렴.
- 작은 $c$: 비선형 압축 강화 → 큰 크기 항에서 곱셈적 성질이 강해짐.
- $c$는 노이즈가 가법적에서 곱셈적으로 전이되는 크기 스케일로 해석한다.

---

## 6. 구현 노트

- 모든 블록은 `forward`에서 `self.training`에 따라 분기한다. 평가 모드
  (`module.eval()`)에서는 노이즈를 더하지 않고 입력을 그대로 반환한다.
- Dropout과 달리 **rescale(예: $1/(1-p)$ 보정)을 하지 않는다.** 가법 노이즈의
  기댓값은 입력과 동일하므로($\mathbb{E}[\varepsilon] = 0$) 별도 보정이 필요 없다.
- `GaussianNoise`의 complex 케이스는 실수부·허수부에 각각 독립 표준 정규
  노이즈를 더한 뒤 입력 dtype으로 캐스팅한다.
- `ArsinhGaussianNoise`의 $c$는 노이즈가 가법적에서 곱셈적 거동으로 전이되는
  크기 스케일을 정한다. `sigma == 0.0`이거나 `training=False`이면 identity로
  단락(short-circuit)된다.

---

## 7. 참고문헌

1. Bishop, C. M. (1995). "Training with Noise is Equivalent to Tikhonov
   Regularization." *Neural Computation*, 7(1), 108–116.
2. Sietsma, J., & Dow, R. J. F. (1991). "Creating artificial neural networks
   that generalize." *Neural Networks*, 4(1), 67–79.
3. arsinh(역 hyperbolic sine) 변환 일반론 — 큰 크기를 로그적으로 압축하는
   부호 보존 변환.
