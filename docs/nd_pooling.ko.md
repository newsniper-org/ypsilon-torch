# N-dimensional Pooling 블록

> English version: [nd_pooling.en.md](nd_pooling.en.md)

## 목차

1. [개요](#1-개요)
2. [수학적 정의](#2-수학적-정의)
3. [아키텍처 및 API 레퍼런스](#3-아키텍처-및-api-레퍼런스)
4. [사용 예제](#4-사용-예제)
5. [하이퍼파라미터 가이드](#5-하이퍼파라미터-가이드)
6. [구현 노트](#6-구현-노트)

---

## 1. 개요

`ypsilon_torch.blocks.pooling`은 **1D/2D/3D를 통합 지원** 하는 robust pooling
블록을 제공한다. `nn.AvgPool{1,2,3}d`나 `nn.MaxPool{1,2,3}d`를 대체하여
outlier에 강건한 풀링을 수행한다.

각 블록은 **L_p 특화 버전** 과 **임의 거리 함수를 받는 general 버전** 의
쌍으로 제공되며, real 및 complex 도메인 모두 지원한다.

| 카테고리 | Real 블록 | Complex 블록 |
|----------|-----------|--------------|
| arsinh 평균 | `AsinhAvgPoolND` | `ComplexAsinhAvgPoolND` |
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

## 2. 수학적 정의

### 공통 표기

- 입력 $\mathbf{X} \in \mathbb{R}^{B \times C \times S_1 \times \cdots \times S_m}$
  (또는 $\mathbb{C}$).
- 출력 $\mathbf{Y} \in \mathbb{R}^{B \times C \times O_1 \times \cdots \times O_m}$.
- $\Omega(o)$: 출력 위치 $o$에 대응하는 local window.

### 풀링 연산

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

### Complex Asinh 풀링의 위상 처리

Complex 도메인의 Asinh 계열은 **magnitude** 에 arsinh 통계를 적용하고,
**위상(phase)** 은 circular mean으로 결정한다:

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

## 3. 아키텍처 및 API 레퍼런스

### 공통 파라미터

모든 pooling 블록이 공유하는 파라미터:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `kernel_size` | `int \| tuple[int, ...]` | 풀링 윈도우 크기 |
| `stride` | `int \| tuple \| None` | 스트라이드. `None`이면 `kernel_size`와 동일 |
| `padding` | `int \| tuple` | 제로 패딩 |

공간 차원 수는 입력 텐서의 `ndim - 2`로 자동 추론된다.

### Real

```python
from ypsilon_torch.blocks.pooling.real import (
    # Asinh (학습 파라미터 있음)
    AsinhAvgPoolND,
    AsinhRMSPoolND,
    # L_p 특화
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

#### Asinh 계열

```python
AsinhAvgPoolND(channels, kernel_size, stride=None, padding=0,
               alpha_init=1.0, dtype_idx=64)
AsinhRMSPoolND(channels, kernel_size, stride=None, padding=0,
               alpha_init=1.0, dtype_idx=64)
```

| 추가 파라미터 | 설명 |
|----------------|------|
| `channels` | 입력 채널 수. 채널별 학습 가능 $\alpha_c$ 생성 |
| `alpha_init` | $\alpha$ 초기값 |

#### L_p 특화

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

| 추가 파라미터 | 설명 |
|----------------|------|
| `dist_fn` | `Callable[[Tensor], Tensor]`. 거리 함수 $\rho$ |

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

시그니처는 real 버전과 동일. Complex Asinh 계열은 `channels` 파라미터를 받는다.

---

## 4. 사용 예제

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

## 5. 하이퍼파라미터 가이드

### `alpha` (Asinh 계열)

- 큰 $\alpha$: arsinh가 거의 선형 → `nn.AvgPool`과 유사하게 동작.
- 작은 $\alpha$: 비선형 압축 → outlier에 강건.
- 채널별 학습 가능하므로 네트워크가 스스로 적응.

### `p` (L_p 특화)

| $p$ | 풀링 특성 |
|-----|-----------|
| 1.0 | 완전한 median pooling (outlier에 가장 강건) |
| 2.0 | average pooling과 동일 |
| 1.5 | 중간 지점 |

### `dist_fn` (General)

```python
# 사전 정의된 거리 함수 예시
l1      = lambda r: r                                  # L1 (median)
l2_sq   = lambda r: r ** 2                             # L2 squared (mean)
huber   = lambda r: torch.where(r < 1, 0.5*r**2, r-0.5)
logcosh = lambda r: torch.log(torch.cosh(r))
```

---

## 6. 구현 노트

### Window 추출

`NDPoolBase._extract_windows`는 `Tensor.unfold()`를 spatial 차원마다
체이닝하여 local window를 추출한다:

```
(B, C, *spatial) → (B, C, *out_spatial, K)
```

$K = \prod_i k_i$ 는 kernel 원소 수. 이 방식으로 1D/2D/3D를 단일 코드로
처리한다.

### Complex Asinh — magnitude + circular mean

Complex arsinh의 branch cut 문제를 우회하기 위해:

1. **Magnitude**: $|z|$에 `sinh_mean_arsinh` 또는 `sinh_rms_arsinh` 적용.
2. **Phase**: window 내 원소들의 circular mean으로 결정.
3. `torch.polar(mag, phase)`로 complex 출력 재구성.

### Medoid — 미분 불가

모든 medoid 계열은 결과가 `.detach()` 상태이다.
