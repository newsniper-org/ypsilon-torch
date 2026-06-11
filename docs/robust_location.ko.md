# Robust 위치·스케일 추정 (확장)

> English version: [robust_location.en.md](robust_location.en.md)

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

`ypsilon_torch.functional.robust_location` 모듈은 **고전적 robust 위치·스케일
추정자** 를 함수형(functional) 인터페이스로 제공한다. 이는
[`robust_stats`](robust_stats.ko.md) 모듈(Fréchet median/medoid, arsinh 공간
통계량)을 M-estimator 및 순서통계량(order-statistic) 기반 추정자로 보완한다.

- **Huber location**: 작은 잔차에서는 이차(quadratic), 큰 잔차에서는
  선형(linear)으로 동작하는 볼록(convex) M-estimator. 영향력(influence)이
  유계(bounded)이다.
- **Tukey biweight location**: redescending M-estimator. tuning 상수를 넘는
  잔차는 영향력이 **0** 이 되어 극단적 outlier를 완전히 기각한다.
- **Trimmed / Winsorized mean**: 가장 극단적인 표본 일부를 제거하거나 clamp 하는
  순서통계량 기반 추정자.
- **Median absolute deviation (MAD)** 및 **biweight midvariance**: robust 스케일
  추정자.

또한 이 추정자들을 normalization 블록으로 감싼 **`HuberLayerNorm`** 과
**`TrimmedLayerNorm`** 을 제공한다.

모든 함수는 단일 `dim` 을 따라 동작하며, 해당 차원이 리덕션된 텐서를 반환한다
(`keepdim=False` 기본). **real 텐서만 지원** 한다.

---

## 2. 수학적 정의

### 2.1 Median absolute deviation (MAD)

$$
\operatorname{MAD}_s(x)
= s \cdot \operatorname{med}_j \bigl\lvert x_j - \operatorname{med}(x) \bigr\rvert
$$

기본 스케일 상수 $s = 1.4826$ 은 가우시안 데이터에서 MAD가 표준편차 $\sigma$ 의
일치(consistent) 추정자가 되도록 하는 상수이다. $s = 1.0$ 은 raw MAD를 준다.

### 2.2 Huber location

Huber 손실 $\rho_\delta$ 를 최소화하는 M-estimator:

$$
\mu_H = \operatorname*{arg\,min}_\mu \sum_j \rho_\delta\!\left(\frac{x_j - \mu}{s}\right),
\qquad s = \operatorname{MAD}(x)
$$

IRLS로 풀며, weight는

$$
w_j = \min\!\left(1,\; \frac{\delta}{\lvert r_j \rvert}\right),
\qquad r_j = \frac{x_j - \mu}{s} .
$$

$\delta$ 는 MAD 스케일 잔차 단위의 tuning 상수이며, 기본 $1.345$ 는 가우시안
잡음에서 약 95% 효율을 준다.

### 2.3 Tukey biweight (bisquare) location

Redescending M-estimator: $\lvert r \rvert > c$ 인 잔차는 weight가 **0** 이 되어
극단적 outlier가 완전히 기각된다.

$$
w_j = \left(1 - (r_j/c)^2\right)^2 \cdot \mathbb{1}\bigl[\lvert r_j \rvert \le c\bigr],
\qquad r_j = \frac{x_j - \mu}{s} .
$$

$c$ 는 MAD 스케일 잔차 단위의 tuning 상수이며, 기본 $4.685$ 는 가우시안 잡음에서
약 95% 효율을 준다.

### 2.4 Trimmed / Winsorized mean

`dim` 을 따라 정렬한 뒤 각 꼬리에서 $k = \lfloor \tau \cdot n \rfloor$ 개의 표본을
처리한다.

$$
\operatorname{TrimmedMean}_\tau(x)
= \frac{1}{n - 2k} \sum_{j = k+1}^{n-k} x_{(j)}
$$

trimmed mean은 양 꼬리에서 $k$ 개를 **제거** 한 뒤 평균을 낸다. winsorized mean은
제거 대신 가장 가까운 보존된 순서통계량으로 **clamp** 한다:

$$
\operatorname{WinsorizedMean}_\tau(x)
= \frac{1}{n} \sum_{j=1}^{n}
  \operatorname{clamp}\bigl(x_{(j)},\, x_{(k+1)},\, x_{(n-k)}\bigr)
$$

$\tau \in [0, 0.5)$ 이다.

### 2.5 Biweight midvariance

robust 분산 추정자:

$$
\zeta_{bi}^2 = \frac{n \sum_{\lvert u_j \rvert < 1} (x_j - M)^2 (1 - u_j^2)^4}
                    {\left(\sum_{\lvert u_j \rvert < 1} (1 - u_j^2)(1 - 5 u_j^2)\right)^2},
\qquad u_j = \frac{x_j - M}{c \cdot \operatorname{MAD}}
$$

여기서 $M$ 은 중앙값이다. 반환값은 이미 midvariance(분산)이므로, biweight
mid-표준편차가 필요하면 `sqrt` 를 취한다. 기본 $c = 9.0$.

### 2.6 Normalization 블록

$\mu$ 를 robust 위치, $s = \operatorname{MAD}(x)$ 를 robust 스케일이라 할 때:

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

## 3. API 레퍼런스

모든 함수는 `ypsilon_torch.functional` 에서 import 가능하다.

### 위치 추정자 (M-estimator)

```python
huber_location(x, dim, delta=1.345, max_iter=20, tol=1e-6, eps=1e-12, keepdim=False) -> Tensor
tukey_biweight_location(x, dim, c=4.685, max_iter=20, tol=1e-6, eps=1e-12, keepdim=False) -> Tensor
```

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `x` | `Tensor` | 입력 (real만 지원) |
| `dim` | `int` | 리덕션 차원 |
| `delta` | `float` | Huber tuning 상수 (MAD 스케일 잔차 단위). 기본 `1.345` |
| `c` | `float` | biweight tuning 상수 (MAD 스케일 잔차 단위). 기본 `4.685` |
| `max_iter` | `int` | IRLS 최대 반복 횟수 |
| `tol` | `float` | 수렴 허용 오차 |
| `eps` | `float` | 수치 안정용 하한 |
| `keepdim` | `bool` | 리덕션 차원 유지 여부 |

### 위치 추정자 (순서통계량)

```python
trimmed_mean(x, dim, trim=0.1, keepdim=False) -> Tensor
winsorized_mean(x, dim, trim=0.1, keepdim=False) -> Tensor
```

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `x` | `Tensor` | 입력 (real만 지원) |
| `dim` | `int` | 리덕션 차원 |
| `trim` | `float` | *각* 꼬리에서 제거/winsorize 하는 비율, `[0, 0.5)`. 기본 `0.1` |
| `keepdim` | `bool` | 리덕션 차원 유지 여부 |

### 스케일 추정자

```python
median_abs_deviation(x, dim, scale=1.4826, keepdim=False) -> Tensor
biweight_midvariance(x, dim, c=9.0, eps=1e-12, keepdim=False) -> Tensor
```

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `x` | `Tensor` | 입력 (real만 지원) |
| `dim` | `int` | 리덕션 차원 |
| `scale` | `float` | MAD 일치 상수. 기본 `1.4826` (가우시안 일치). raw MAD는 `1.0` |
| `c` | `float` | biweight midvariance tuning 상수. 기본 `9.0` |
| `eps` | `float` | 수치 안정용 하한 |
| `keepdim` | `bool` | 리덕션 차원 유지 여부 |

### Normalization 블록

```python
from ypsilon_torch.blocks.normalizations.real import HuberLayerNorm, TrimmedLayerNorm

HuberLayerNorm(normalized_shape, delta=1.345, eps=1e-5, dim=-1,
               affine=True, dtype_idx=64, max_iter=10)
TrimmedLayerNorm(normalized_shape, trim=0.1, eps=1e-5, dim=-1,
                 affine=True, dtype_idx=64)
```

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `normalized_shape` | `int` | normalize 할 차원의 크기 |
| `delta` | `float` | (Huber) Huber tuning 상수. 기본 `1.345` |
| `trim` | `float` | (Trimmed) 각 꼬리에서 제거하는 비율, `[0, 0.5)`. 기본 `0.1` |
| `eps` | `float` | 스케일에 더하는 안정용 epsilon. 기본 `1e-5` |
| `dim` | `int` | normalize 차원. 기본 `-1` |
| `affine` | `bool` | True면 학습 가능한 `gamma`/`beta` 적용. 기본 True |
| `dtype_idx` | `FPDTypeIdx` | 부동소수점 정밀도 인덱스. 기본 `64` (`get_float_dtype` 로 dtype 결정) |
| `max_iter` | `int` | (Huber) IRLS 최대 반복 횟수. 기본 `10` |

---

## 4. 사용 예제

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

# robust 스케일 (MAD, 가우시안 일치)
s = median_abs_deviation(x, dim=-1)                 # shape: (8,)

# Huber location (유계 영향력)
mu_h = huber_location(x, dim=-1, delta=1.345)       # shape: (8,)

# Tukey biweight location (redescending)
mu_t = tukey_biweight_location(x, dim=-1, c=4.685)  # shape: (8,)

# trimmed / winsorized mean
mu_tr = trimmed_mean(x, dim=-1, trim=0.1)           # shape: (8,)
mu_wn = winsorized_mean(x, dim=-1, trim=0.1)        # shape: (8,)

# biweight midvariance (robust 분산)
var = biweight_midvariance(x, dim=-1, c=9.0)        # shape: (8,)
```

Normalization 블록:

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

## 5. 하이퍼파라미터 가이드

| 추정자 | 상수 | 기본값 | 의미 |
|--------|------|--------|------|
| `huber_location` | `delta` | `1.345` | 가우시안에서 ~95% 효율. 작을수록 더 robust(영향력 더 빨리 절단), 클수록 평균에 근접 |
| `tukey_biweight_location` | `c` | `4.685` | 가우시안에서 ~95% 효율. $\lvert r \rvert > c$ 잔차는 완전 기각. 작을수록 공격적으로 outlier 제거 |
| `trimmed_mean` / `winsorized_mean` | `trim` | `0.1` | 각 꼬리에서 다루는 비율. `0`이면 일반 평균, `0.5`에 가까울수록 중앙값에 근접 |
| `median_abs_deviation` | `scale` | `1.4826` | 가우시안 일치 상수. raw MAD는 `1.0` |
| `biweight_midvariance` | `c` | `9.0` | midvariance tuning 상수 (관례값) |

선택 가이드:

- **유계이지만 비제로(non-zero) 영향력** 이 필요하고 볼록성(수렴 보장)을 원하면
  → Huber.
- **극단 outlier 완전 기각** (redescending)을 원하면 → Tukey biweight. 단,
  비볼록이라 초기값에 민감할 수 있다(MAD/median 초기화로 완화).
- **단순하고 빠른 breakdown-유계 추정** 이 필요하면 → trimmed/winsorized mean
  (정렬 한 번).

---

## 6. 구현 노트

### IRLS (Huber / Tukey biweight)

두 M-estimator 모두 동일한 IRLS 구조를 공유한다:

1. $\mu^{(0)} = \operatorname{median}(x)$, $s = \operatorname{MAD}(x)$ (둘 다
   `eps` 로 하한 clamp) 로 초기화.
2. $r = (x - \mu)/s$ 로 잔차를 구하고, 추정자별 weight $w_j$ 를 계산한다.
3. $\mu^{(k+1)} = \dfrac{\sum_j w_j x_j}{\sum_j w_j}$.
4. 수렴($\lvert \mu^{(k+1)} - \mu^{(k)} \rvert$ 최댓값 $< $ `tol`) 또는
   `max_iter` 도달 시 종료.

IRLS 루프는 `torch.no_grad()` 안에서 실행된다. 즉 반환된 위치 $\mu$ 는 detach된
상수처럼 동작하며, gradient는 정규화 식의 $(x - \mu)/s$ 경로로만 흐른다. 이는
[`robust_stats`](robust_stats.ko.md) 의 `frechet_median_lp` 와 **동일한
컨벤션** 이다(IRLS 반복은 grad를 전파하지 않고, 위치는 detach된 상수로 취급).

### 순서통계량 (trimmed / winsorized)

`torch.sort` 로 정렬 후, trimmed mean은 `narrow` 로 양 꼬리 $k$ 개를 잘라낸 뒤
평균을 낸다. winsorized mean은 `torch.minimum`/`torch.maximum` 으로 경계
순서통계량에 clamp 한다. IRLS와 달리 단일 정렬의 closed-form이며 보존된 표본을
통해 완전히 미분 가능하다.

### Normalization 블록의 grad 컨벤션

`HuberLayerNorm` 은 위치로 `huber_location`, 스케일로 `median_abs_deviation` 을
사용한다. `TrimmedLayerNorm` 은 위치로 `trimmed_mean` 을 사용한다. 두 블록 모두
정규화는 $(x - \mu)/(s + \epsilon) \cdot \gamma + \beta$ 이다.

Huber 위치는 IRLS(`no_grad`)로 풀리므로 detach된 상수로 동작하며, gradient는
robust_stats의 `frechet_median_lp` 와 같은 컨벤션으로 $(x - \mu)/s$ 경로로만
흐른다. Trimmed 위치는 보존된 표본을 통해 미분 가능하다. 두 블록 모두 MAD 스케일
역시 median 연산을 거치므로 사실상 detach된 상수로 동작한다.

`dtype_idx`(`FPDTypeIdx`, 기본 `64`)는 `get_float_dtype` 을 통해 학습 파라미터
`gamma`/`beta` 의 부동소수점 정밀도를 결정한다.

이 블록들은 영향력이 유계인 Huber 위치 덕분에 평균 기반 `AsinhMeanLayerNorm` 과
IRLS 기반 `RobustLayerNorm` 사이에 위치한다.

---

## 7. 참고문헌

1. Huber, P. J. (1964). "Robust Estimation of a Location Parameter."
   *Annals of Mathematical Statistics*, 35(1), 73–101.
2. Tukey, J. W. — biweight (bisquare) redescending M-estimator.
   Beaton, A. E., & Tukey, J. W. (1974). "The Fitting of Power Series,
   Meaning Polynomials, Illustrated on Band-Spectroscopic Data."
   *Technometrics*, 16(2), 147–185.
3. Hampel, F. R. (1974). "The Influence Curve and Its Role in Robust
   Estimation." (MAD as a robust scale estimator.)
4. [Robust gradient estimation](https://opt-ml.org/papers/2025/paper74.pdf)
