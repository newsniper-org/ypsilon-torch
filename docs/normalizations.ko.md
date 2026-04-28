# Normalization 블록

> English version: [normalizations.en.md](normalizations.en.md)

## 목차

1. [개요](#1-개요)
2. [수학적 정의](#2-수학적-정의)
3. [아키텍처 및 API 레퍼런스](#3-아키텍처-및-api-레퍼런스)
4. [사용 예제](#4-사용-예제)
5. [하이퍼파라미터 가이드](#5-하이퍼파라미터-가이드)
6. [구현 노트](#6-구현-노트)
7. [참고문헌](#7-참고문헌)

---

## 1. 개요

`ypsilon_torch.blocks.normalizations`는 outlier에 강건한 정규화 블록을
제공한다. Real 및 complex 텐서 모두 지원하며, 기존
`ComplexLayerNorm`/`ComplexRMSNorm` 외에 다음을 추가한다:

| 블록 | 도메인 | 핵심 아이디어 |
|------|--------|---------------|
| `RobustLayerNorm` | Real | L_φ 위치 추정 + MAD 스케일 |
| `AsinhMeanLayerNorm` | Real | arsinh 변환 공간에서 정규화 |
| `ComplexRobustLayerNorm` | Complex | Weiszfeld 위치 추정 + MAD 스케일 |
| `ComplexAsinhMeanLayerNorm` | Complex | magnitude 기반 arsinh 정규화 |

---

## 2. 수학적 정의

### 2.1 RobustLayerNorm

$1 < \phi < 2$ (권장: $1.2 \lesssim \phi < 2$).

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

$c$는 학습 가능한 스칼라 파라미터이다.

### 2.3 ComplexRobustLayerNorm

복소 평면에서 Weiszfeld 알고리즘으로 $\mu_\phi$를 구한다.
스케일 $s_\phi$는 magnitude의 MAD이다 (실수).

$$
\operatorname{ComplexRobustLayerNorm}_\phi(z)_j
= \gamma_j \frac{z_j - \mu_\phi}{s_\phi + \epsilon} + \beta_j
$$

$\gamma, \beta$는 complex 파라미터. 위상(phase)이 자연스럽게 보존된다.

### 2.4 ComplexAsinhMeanLayerNorm

magnitude 기반 접근 (complex arsinh의 branch cut 회피):

$$
m_j = |z_j|, \quad u_j = \operatorname{arsinh}(m_j / c)
$$

$$
\hat{u}_j = \frac{u_j - \mathbb{E}(U)}{\sqrt{\mathbb{V}(U) + \epsilon}}
$$

$$
\operatorname{out}_j = \gamma_j \cdot z_j \cdot \frac{\hat{u}_j}{m_j + \epsilon} + \beta_j
$$

$\gamma, \beta$는 complex 파라미터.

---

## 3. 아키텍처 및 API 레퍼런스

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

| 파라미터 | 설명 |
|----------|------|
| `normalized_shape` | 정규화 차원의 크기 |
| `phi` | L_φ 위치 추정 지수. $(1, 2)$ 범위 |
| `dim` | 정규화 차원. 기본 `-1` |
| `affine` | `True`이면 학습 가능 `gamma`, `beta` 적용 |
| `max_iter` | IRLS 최대 반복 횟수 |

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

| 파라미터 | 설명 |
|----------|------|
| `c_init` | arsinh 스케일 `c`의 초기값 (학습 가능) |

### Complex

```python
from ypsilon_torch.blocks.normalizations.complex import (
    ComplexRobustLayerNorm,
    ComplexAsinhMeanLayerNorm,
    ComplexLayerNorm,        # 기존
    ComplexRMSNorm,          # 리팩터링됨
)
```

#### `ComplexRobustLayerNorm`

`RobustLayerNorm`과 동일한 시그니처. `gamma`, `beta`가 complex 타입.

#### `ComplexAsinhMeanLayerNorm`

`AsinhMeanLayerNorm`과 동일한 시그니처. `gamma`, `beta`가 complex 타입,
`c`는 real 스칼라.

#### `ComplexRMSNorm` (리팩터링)

```python
ComplexRMSNorm(
    normalized_shape: int,
    eps: float = 1e-6,
    dim: int = -1,
    affine: bool = False,
    dtype_idx: FPDTypeIdx = 64,
)
```

기존 대비 변경사항:
- `d_model` → `normalized_shape` 개명
- `dim` 파라미터 추가 (기본 `-1`, 기존 하드코딩)
- `affine` 플래그 추가 (기본 `False`)
- `reset_parameters()` 메서드 추가

---

## 4. 사용 예제

```python
import torch
from ypsilon_torch.blocks.normalizations.real import (
    RobustLayerNorm, AsinhMeanLayerNorm,
)
from ypsilon_torch.blocks.normalizations.complex import (
    ComplexRobustLayerNorm, ComplexAsinhMeanLayerNorm,
)

# Real — Transformer 스타일 (B, seq_len, d_model)
x = torch.randn(2, 128, 256)
rln = RobustLayerNorm(256, phi=1.5)
y = rln(x)                                        # (2, 128, 256)

aln = AsinhMeanLayerNorm(256, c_init=1.0)
y2 = aln(x)                                       # (2, 128, 256)

# Real — CNN 스타일 (B, C, H, W), 채널 dim 정규화
x_cnn = torch.randn(2, 64, 16, 16)
rln_ch = RobustLayerNorm(64, dim=1)
y3 = rln_ch(x_cnn)                                # (2, 64, 16, 16)

# Complex — (B, d_model) complex64
z = torch.randn(2, 128, 256, dtype=torch.complex64)
crln = ComplexRobustLayerNorm(256)
w = crln(z)                                        # (2, 128, 256) complex64

caln = ComplexAsinhMeanLayerNorm(256)
w2 = caln(z)                                       # (2, 128, 256) complex64
```

---

## 5. 하이퍼파라미터 가이드

### `phi` (RobustLayerNorm / ComplexRobustLayerNorm)

| 값 | 특성 |
|----|------|
| $\phi \to 1^+$ | ordinary median에 가까움. 극단적 outlier에 가장 강건 |
| $\phi = 1.5$ | **권장 기본값**. robustness와 효율성의 균형 |
| $\phi \to 2^-$ | mean에 가까움. outlier 민감도 증가 |

참고자료 1에 따르면 $1.2 \lesssim \phi < 2$ 범위가 권장된다.

### `c` (AsinhMeanLayerNorm / ComplexAsinhMeanLayerNorm)

- 큰 $c$: arsinh가 거의 선형 → 표준 LayerNorm에 수렴.
- 작은 $c$: 비선형 압축 효과 강화 → outlier에 더 강건.
- 학습 가능하므로 초기값(`c_init`)만 설정하면 된다.

---

## 6. 구현 노트

- IRLS / Weiszfeld 루프는 `torch.no_grad()` 내에서 실행되어 중간 계산의
  메모리를 절약한다. 최종 정규화 단계만 gradient를 전파한다.
- `dim` 파라미터로 정규화 차원을 자유롭게 설정 가능하다 (`-1` = 마지막 차원,
  `1` = 채널 차원 등).
- `gamma`/`beta`의 broadcast shape는 `dim`에 따라 동적으로 생성된다.

---

## 7. 참고문헌

1. [Robust gradient estimation](https://opt-ml.org/papers/2025/paper74.pdf)
