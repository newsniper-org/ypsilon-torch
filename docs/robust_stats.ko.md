# Robust Statistics (함수형 유틸리티)

> English version: [robust_stats.en.md](robust_stats.en.md)

## 목차

1. [개요](#1-개요)
2. [수학적 정의](#2-수학적-정의)
3. [API 레퍼런스](#3-api-레퍼런스)
4. [사용 예제](#4-사용-예제)
5. [구현 노트](#5-구현-노트)
6. [참고문헌](#6-참고문헌)

---

## 1. 개요

`ypsilon_torch.functional.robust_stats` 모듈은 **Fréchet median/medoid** 및
**arsinh 공간 통계량** 을 함수형(functional) 인터페이스로 제공한다.

- **Fréchet median**: 임의 거리 함수 $\rho$ 아래에서 총 거리를 최소화하는
  연속 위치 추정자.
- **Fréchet medoid**: 표본 집합 내에서 총 거리를 최소화하는 이산 위치 추정자.
- **arsinh 공간 통계**: $\operatorname{arsinh}$ 변환 공간에서 평균/RMS를
  계산한 뒤 $\sinh$로 역변환하는 robust 통계량.

모든 함수는 **real 텐서와 complex 텐서를 모두 지원** 한다 (arsinh 계열 제외).

---

## 2. 수학적 정의

### 2.1 Fréchet median & medoid

거리 함수 $\rho : [0, +\infty) \to [0, +\infty)$ 에 대해,

$$
\operatorname{FréchetMedian}_\rho(X)
= \arg\min_{y \in M} \sum_{x \in X} \rho\!\bigl(\lvert x - y \rvert\bigr)
$$

$$
\operatorname{FréchetMedoid}_\rho(X)
= \arg\min_{y \in X} \sum_{x \in X} \rho\!\bigl(\lvert x - y \rvert\bigr)
$$

**L_p 특수 케이스** ($\rho(r) = r^p$):

| $p$ | Fréchet median | 비고 |
|-----|----------------|------|
| 1   | ordinary median | `torch.median` 직접 사용 |
| 2   | arithmetic mean | `torch.mean` 직접 사용 |
| $(1, 2)$ | IRLS 반복 풀이 | |

Complex 텐서의 경우 거리는 complex modulus $\lvert z_i - z_j \rvert$ 를 사용하며,
Weiszfeld 알고리즘으로 복소 평면 위의 geometric median을 구한다.

### 2.2 arsinh 공간 통계

변환 쌍: $f_c(x) = \operatorname{arsinh}(x/c)$, $g_c(z) = c \sinh(z)$.

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

## 3. API 레퍼런스

모든 함수는 `ypsilon_torch.functional` 에서 import 가능하다.

### General (임의 거리 함수)

```python
frechet_median(x, dim, dist_fn, max_iter=20, tol=1e-6) -> Tensor
frechet_medoid(x, dim, dist_fn) -> Tensor
```

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `x` | `Tensor` | 입력 (real 또는 complex) |
| `dim` | `int` | 리덕션 차원 |
| `dist_fn` | `Callable[[Tensor], Tensor]` | 거리 함수 $\rho$. non-negative 잔차 크기를 받아 거리를 반환. 미분 가능해야 함 (median만) |
| `max_iter` | `int` | IRLS 최대 반복 횟수 |
| `tol` | `float` | 수렴 허용 오차 |

`frechet_medoid`의 반환값은 `.detach()` 상태 (gradient 미전파).

### L_p 특화

```python
frechet_median_lp(x, dim, p=1.0, max_iter=20, tol=1e-6) -> Tensor
frechet_medoid_lp(x, dim) -> Tensor
```

### arsinh 합성

```python
sinh_mean_arsinh(x, c, dim) -> Tensor
sinh_rms_arsinh(x, c, dim) -> Tensor
sinh_frechet_median_arsinh(x, c, dim, dist_fn, ...) -> Tensor
sinh_frechet_median_lp_arsinh(x, c, dim, p=1.0, ...) -> Tensor
sinh_frechet_medoid_arsinh(x, c, dim, dist_fn) -> Tensor
sinh_frechet_medoid_lp_arsinh(x, c, dim) -> Tensor
```

### Root-\*-Square 합성

```python
root_frechet_median_square(x, dim, dist_fn, ...) -> Tensor
root_frechet_median_lp_square(x, dim, p=1.0, ...) -> Tensor
root_frechet_medoid_square(x, dim, dist_fn) -> Tensor
root_frechet_medoid_lp_square(x, dim) -> Tensor
```

---

## 4. 사용 예제

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

# Huber 거리 기반 median
huber = lambda r: torch.where(r < 1.0, 0.5 * r**2, r - 0.5)
med_huber = frechet_median(x, dim=-1, dist_fn=huber)  # shape: (8,)

# arsinh 공간 평균
avg = sinh_mean_arsinh(x, c=1.0, dim=-1)           # shape: (8,)

# Complex geometric median
z = torch.randn(8, 64, dtype=torch.complex64)
gmed = frechet_median_lp(z, dim=-1)                 # shape: (8,), complex
```

---

## 5. 구현 노트

### IRLS (Iteratively Reweighted Least Squares)

L_p median ($1 < p < 2$)의 풀이:

1. $\mu^{(0)} = \operatorname{median}(x)$ 로 초기화.
2. $w_j^{(k)} = \lvert x_j - \mu^{(k)} \rvert^{p-2}$, $\mu^{(k+1)} = \frac{\sum w_j x_j}{\sum w_j}$.
3. 수렴 또는 `max_iter` 도달 시 종료.

IRLS 루프는 `torch.no_grad()` 안에서 실행되어 메모리를 절약한다.

### General distance — autodiff 기반 weight 계산

임의 $\rho$ 에 대한 IRLS weight는:

$$
w_j = \frac{\rho'(r_j)}{r_j}, \qquad r_j = \lvert x_j - \mu \rvert
$$

$\rho'$는 `torch.autograd.grad` 로 자동 계산된다 (`torch.enable_grad()` 로
`no_grad` 블록 내에서 국소적으로 활성화).

### Weiszfeld (complex)

복소 평면에서의 geometric median 풀이. 실수 IRLS와 동일한 구조이나,
$w_j = 1 / \lvert z_j - \mu \rvert$ (L1 케이스) 또는 autodiff weight
(general 케이스)를 사용한다.

### Medoid — 미분 불가

`frechet_medoid`는 이산 `argmin` 이므로 결과에 `.detach()`가 적용된다.

---

## 6. 참고문헌

1. Weiszfeld, E. (1937). "Sur le point pour lequel la somme des distances de n
   points donnés est minimum."
2. [Robust gradient estimation](https://opt-ml.org/papers/2025/paper74.pdf)
