# Mixture (희소·조건부 연산) 블록

> English version: [mixture.en.md](mixture.en.md)

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

`ypsilon_torch.blocks.mixture`는 **희소(sparse) / 조건부(conditional) 연산**
블록 5종을 모은다. 각 블록은 "모든 토큰에 같은 양의 연산을 쓰지 않는다"는
공통 아이디어를 서로 다른 축에서 구현한다.

| 블록 | 무엇을 라우팅하는가 | 라우팅 방식 | aux loss |
|---|---|---|---|
| `MixtureOfDepths` (MoD) | 토큰을 블록에 보낼지 말지 (깊이) | expert-choice top-k | predictor BCE |
| `MixtureOfDepthsAndExperts` (MoDE) | 토큰을 어느 expert(또는 no-op)로 | token-choice top-1 | Switch load-balance |
| `MixtureOfHiddenDimensions` (MoHD) | hidden 차원의 sub-dim 그룹 | shared + top-k | Switch load-balance |
| `MixtureOfLookupExperts` (MoLE) | 토큰 임베딩에 대한 dense expert | dense softmax (top-k 없음) | 없음 |
| `AttractorPatchNetwork` (APN) | 프로토타입(attractor) 패치 | cosine top-k | usage uniform-balance |

aux loss를 내는 블록은 모두
`ypsilon_torch.blocks.AuxLossModule`을 상속한다. 이는
[Bilinear MLP 문서](bilinear_mlp.ko.md)의 `BilinearMLPBase`와 **완전히 동일한**
`reg_loss` 메커니즘(`torchutils.auxloss` 기반)을 쓴다. 따라서
`AuxLossModule.reg_loss.collect(model)` **한 번**으로 모델 안의 mixture 블록과
bilinear 블록의 손실이 함께 수집된다(검증됨, §4.6).

### 데이터 흐름 (블록별 한눈에)

```
MoD   x (B,N,d) ─► router r=w·x ─► top-C 토큰만 ─► f(X̃) ─► r·f + 잔차 ─► (B,N,d)
                                  (나머지 토큰은 잔차로 우회)

MoDE  (integrated) x ─► token-choice top-1 라우터 [E experts + 1 no-op] ─► x + g·expert(x)
      (staged)     x ─► MoD 토큰 선별 ─► 별도 MoE 라우터 ─► expert(x)

MoHD  x ─► W_up ─► N개 sub-dim 그룹 ─► [shared 항상 on | specialized top-k]
        ─► α 스케일 ─► grouped fusion ─► SiLU ─► W_down ─► (B,N,d)

MoLE  router(h) ─┐
      e (임베딩) ─► FFN_j(e) ─► Σ g_j FFN_j(e)  ──► h + shared(h) + Σ ─► (B,N,d)
      (추론: FFN_j(e)를 vocab 전체에 대해 LUT로 사전계산 → 인덱싱)

APN   h ─► LN ─► cosine(프로토타입 K개)/τ ─► top-k, softmax w
        ─► 공유 코드 u=Vᵀz, φ_i=u⊙σ(a_i⊙u+b_i), Δ_i=U_i φ_i
        ─► h + γ·Σ w_i Δ_i ─► (B,N,d)
```

---

## 2. 수학적 정의

표기: 입력 `x ∈ ℝ^{B×N×d}` (배치 `B`, 시퀀스 `N`, 차원 `d`). `f`는 래핑된
블록, `r_i`는 토큰 `i`의 라우터 점수, `g`는 게이트(softmax) 가중치.

### 2.1 Mixture-of-Depths (MoD)

라우터는 토큰별 스칼라 점수 `r_i = w_θ^⊤ x_i`를 낸다. 용량
`C = ⌊capacity_ratio · N⌋` 만큼의 토큰을 **expert-choice top-k** 로 선택한다
(블록이 자신이 처리할 top-`C` 토큰을 고른다). 선택 집합을
`X̃ = TopC_i(r_i)`라 하면:

$$
x_i^{l+1} = \begin{cases}
    r_i\, f(\tilde X)_i + x_i & r_i > P_\beta(R) \\
    x_i & \text{otherwise}
\end{cases}
\qquad \beta = 1 - C/N,
$$

여기서 `P_β(R)`는 점수 분포 `R`의 `β`-분위수(즉 top-`C` 경계). 선택된 토큰만
`r_i·f(X̃)_i`만큼 잔차에 더해지고, 미선택 토큰은 항등(identity)으로
통과한다.

**load-balance loss가 불필요한 이유.** expert-choice 라우팅에서는 블록이 정확히
`C`개 토큰만 고르므로 부하가 구조적으로 균등하다 — token-choice MoE에서
필요한 load-balancing loss가 필요 없다.

**predictor (자기회귀 추론용).** 시퀀스 전체에 대한 top-`k`는 **non-causal**
(미래 토큰을 봄)이라 생성 시 그대로 쓸 수 없다. 그래서 입력에 stop-gradient를
건 보조 **predictor MLP**를 top-`k` 멤버십에 대한 BCE로 학습한다:

$$
\mathcal{L}_{\text{aux}}
= \mathrm{BCE}\bigl(\text{predictor}(\mathrm{sg}[x]),\;
  \mathbb{1}[r \ge \text{threshold}]\bigr),
\qquad
\text{threshold} = r_{(C)}.
$$

이 `reg_loss = aux_weight · BCE`가 학습 손실에 더해지고, 추론 시에는
`predict_route(x) = (predictor(x) > 0)`로 토큰별·인과적으로 라우팅한다.

### 2.2 Mixture-of-Depths-and-Experts (MoDE)

arXiv:2404.02258 §4. MoD의 깊이 라우팅과 sparse MoE를 결합한다. 두 변형:

**integrated (논문 권장).** 실제 expert `N`개에 **no-op(identity) expert** 1개를
더해 총 `N+1` 슬롯에 대한 **단일 token-choice top-1 라우터**를 쓴다. 토큰이
no-op 슬롯으로 라우팅되는 것이 곧 MoD의 우회(bypass)이므로, 깊이 선택과 expert
선택이 하나의 라우팅 연산을 공유한다. `g = softmax(W_r x)`, `(g*, e*) = max_e g`:

$$
y_i = x_i + g^*_i \cdot \mathrm{FFN}_{e^*_i}(x_i)
\quad (\text{no-op 슬롯이면 } \mathrm{FFN} = 0).
$$

**staged.** MoD 라우터가 먼저 처리할 토큰을 선별하고(§2.1), 살아남은 토큰을
별도 MoE 라우터가 expert에 배정한다(`residual=False`).

**Switch-style load-balance loss** (모든 슬롯, no-op 포함):

$$
\mathcal{L}_{\text{bal}}
= \alpha \cdot S \cdot \sum_{s=1}^{S} f_s\, p_s,
\qquad
f_s = \tfrac{1}{BN}\!\sum_{i} \mathbb{1}[e^*_i = s],
\quad
p_s = \tfrac{1}{BN}\!\sum_{i} g_{i,s},
$$

여기서 `S`는 슬롯 수(`N` 또는 `N+1`), `f_s`는 슬롯 `s`로 디스패치된 토큰 비율,
`p_s`는 슬롯 `s`의 평균 라우팅 확률, `α = aux_weight`.

### 2.3 Mixture-of-Hidden-Dimensions (MoHD)

arXiv:2412.05644. MoE가 FFN을 **expert 서브네트워크**로 쪼개는 대신, MoHD는
**hidden 차원** `d_ff`를 `N`개 sub-dim 그룹(각 크기 `d_e = d_ff/N`)으로 쪼개고
토큰별로 일부 그룹만 켠다. 라우터는 centroid `φ_i`로 affinity를 낸다:

$$
s = \mathrm{softmax}(x\, C^\top) \in \mathbb{R}^{N_{\text{sub}}},
\qquad C \in \mathbb{R}^{N_{\text{sub}}\times d}.
$$

앞쪽 `shared`개 그룹은 **항상 on**, 나머지 `specialized` 그룹에서 `top-k`를
고른다(`k = active − shared`). 게이트 `g`는 shared 그룹엔 `s`를 그대로,
specialized 그룹엔 top-k 위치에만 `s`를 둔다.

게이트로 인한 활성 크기 손실은 **per-token 스케일 `α`** 로 복원한다:

$$
\alpha = \Bigl(\sum_i g_i\Bigr)\cdot N_{\text{sub}},
\qquad
h = \alpha \cdot \bigl(g \odot (W_{\text{up}} x)\bigr).
$$

이어서 **grouped fusion**(블록대각 linear)으로 그룹 내부를 재혼합하고, SiLU
활성화 후 down-proj 한다:

$$
y = W_{\text{down}}\,\mathrm{SiLU}\bigl(\mathrm{Fusion}(h)\bigr).
$$

> grouped fusion은 원논문의 **Monarch 행렬**(`O(d/r)` 비용)을 그대로 구현한 것이
> 아니라, 같은 "값싼 구조적 재혼합" 역할을 하는 **블록대각 grouped linear**로
> 대체한 것이다. `fusion=False`로 끌 수 있다.

**load-balance loss** (specialized 그룹에 대한 Switch-style):
`f`는 그룹별 선택 빈도, `p`는 그룹별 평균 affinity, `n_spec`는 specialized
그룹 수일 때 `L = α · n_spec · Σ f_g p_g`.

### 2.4 Mixture-of-Lookup-Experts (MoLE)

arXiv:2503.15798. **핵심 비대칭** — 라우터는 문맥적 hidden 상태 `h`를 읽지만,
각 **routed expert는 토큰 임베딩 `e`(토큰 id의 함수)만** 읽는다. 라우팅은
**dense**(전체 expert에 대한 softmax, top-k 아님)이라 load-balance loss가
필요 없다:

$$
g = \mathrm{softmax}(h\, W_r^\top), \qquad
h' = h + \mathrm{FFN}_{\text{shared}}(h) + \sum_{j=1}^{N} g_j\, \mathrm{FFN}_j(e).
$$

shared expert는 `h`에 의존해 완전한 문맥 용량을 유지한다.

**LUT 재매개화 (추론).** 각 routed expert의 출력 `FFN_j(e)`는 토큰 id의
함수이므로, 학습 후 **vocab 전체에 대해** 사전계산해 LUT
`∈ ℝ^{V×N×d}`로 만들 수 있다:

$$
\mathrm{LUT}[v, j, :] = \mathrm{FFN}_j\bigl(\mathrm{Embedding}(v)\bigr).
$$

추론 시에는 expert FFN을 전혀 돌리지 않고 토큰 id로 LUT를 인덱싱한다:

$$
h' = h + \mathrm{FFN}_{\text{shared}}(h)
     + \sum_j g_j\, \mathrm{LUT}[\text{id}, j, :].
$$

이는 근사가 아니라 **정확히 동일**하다 (검증됨: train vs LUT maxdiff `0.0`,
§6).

### 2.5 Attractor Patch Networks (APN)

arXiv:2602.06993. dense FFN을 **조각별 저랭크(piecewise low-rank) 잔차** 군으로
대체한다. 토큰 `h`를 LN한 `z`와 `K`개 프로토타입(attractor) `p_i`의 코사인
유사도를 온도 `τ`로 나눠 점수화, top-`k`를 고르고 softmax로 가중한다:

$$
s_i = \frac{\langle \mathrm{LN}(h),\, \widehat{p_i}\rangle}{\tau},
\qquad
\mathcal{K} = \mathrm{TopK}_i(s_i, k),
\qquad
w_i = \mathrm{softmax}_{i\in\mathcal{K}}(s_i).
$$

모든 패치가 공유하는 **compact code** `u`(차원 `r ≪ d`)에서, 패치별 게이트로
저랭크 잔차 `Δ_i`를 만든다:

$$
u = V^\top \mathrm{LN}(h) \in \mathbb{R}^r,
\qquad
\varphi_i = u \odot \sigma(a_i \odot u + b_i),
\qquad
\Delta_i = U_i\, \varphi_i,
$$

$$
y = h + \gamma \sum_{i\in\mathcal{K}} w_i\, \Delta_i.
$$

**load-balance loss.** 패치 사용 빈도(usage)가 균등 분포 `1/K`에 가깝도록:

$$
\mathcal{L}_{\text{bal}}
= \alpha \sum_{i=1}^{K} \Bigl(\mathrm{usage}_i - \tfrac{1}{K}\Bigr)^2,
\qquad
\mathrm{usage}_i = \frac{\#\{\text{top-k에 } i \text{가 뽑힌 횟수}\}}{\sum_j \#\{\cdots\}}.
$$

> **주의:** 기본 하이퍼파라미터(`K=256`, `r=32`, `τ=0.07`, `k∈[2,8]`)는 소스에서
> 자동 조사(automated reader)로 추출돼 검증 플래그가 붙은 값이다. 그대로 동작은
> 하지만, 원논문 수치와 대조 검증한 뒤 운영에 쓰기를 권장한다.

---

## 3. 아키텍처 및 API 레퍼런스

### 3.0 `AuxLossModule(torch.nn.Module)`

aux loss를 내는 모든 mixture 블록의 베이스. `BilinearMLPBase`와 **동일한**
`reg_loss` 메커니즘을 `torchutils.auxloss` 위에 둔 standalone 베이스다.

- 게터: `self.reg_loss` → 현재 모듈의 `_reg_loss` 스칼라 tensor.
- 세터: `self.reg_loss = loss` (Tensor 또는 `float`). dtype/shape 정규화.
- 수집기(**static-style 호출**): `AuxLossModule.reg_loss.collect(model)` —
  `model.modules()`를 순회하며 모든 `_reg_loss`를 `torch.sum`으로 집계. mixture
  블록과 bilinear 블록을 **함께** 모은다.
- 리셋터: `AuxLossModule.reg_loss.reset(model, 0.0)` 또는 `model.reg_loss = 0.0`.

`_reg_loss`는 non-persistent buffer라 `.to(device/dtype)`는 따라가지만
`state_dict`에는 들어가지 않는다.

### 3.1 `MixtureOfDepths(AuxLossModule)`

```python
MixtureOfDepths(
    block: nn.Module,
    dim: int,
    capacity_ratio: float = 0.125,
    aux_weight: float = 1.0,
    predictor_hidden: int | None = None,
)
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `block` | — | 래핑되는 블록. `(B, C, d) → (B, C, d)`. 선택된 토큰만 본다. |
| `dim` | — | 토큰 차원 `d` |
| `capacity_ratio` | `0.125` | 처리 토큰 비율 `C/N` (논문 권장 12.5%) |
| `aux_weight` | `1.0` | predictor BCE 가중치. `0.0`이면 손실 기여만 끔(predictor는 여전히 추론 가능) |
| `predictor_hidden` | `None`(=`dim`) | 인과 predictor MLP의 hidden 폭 |

- `forward(x) -> (B, N, d)`: expert-choice top-`C` 라우팅. 선택 토큰은
  원래 순서로 정렬(causal 블록 호환). 출력은 `scatter_add_`로 잔차에 더함.
- `predict_route(x) -> bool (B, N)`: 자기회귀 추론용 인과 라우팅
  마스크(`predictor(x) > 0`).
- 서브모듈: `router = nn.Linear(dim, 1, bias=False)`, `predictor =
  Sequential(Linear(dim,ph), GELU, Linear(ph,1))`.

### 3.2 `MixtureOfDepthsAndExperts(AuxLossModule)`

```python
MixtureOfDepthsAndExperts(
    dim: int,
    num_experts: int = 4,
    variant: Literal["integrated", "staged"] = "integrated",
    capacity_ratio: float = 0.125,
    expert_hidden: int | None = None,
    expansion: float = 4.0,
    aux_weight: float = 1e-2,
)
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `dim` | — | 토큰 차원 |
| `num_experts` | `4` | 실제 expert 수 `N` (integrated는 no-op 슬롯 1개 추가) |
| `variant` | `"integrated"` | `"integrated"`(논문 권장) / `"staged"` |
| `capacity_ratio` | `0.125` | MoD 단계 토큰 비율 (`"staged"`에서만 사용) |
| `expert_hidden` | `None`(=`expansion·dim`) | expert FFN 폭 |
| `expansion` | `4.0` | `expert_hidden` 미지정 시 폭 배수 |
| `aux_weight` | `1e-2` | MoE load-balance(및 staged의 MoD predictor) 가중치 |

- expert는 `SwiGLUFeedForward(dim, ...)`. integrated는 `_Top1MoE`(no-op 포함,
  residual) 하나, staged는 `_Top1MoE`(no-op 없음, residual 없음)를
  `MixtureOfDepths`로 감싼다.

### 3.3 `MixtureOfHiddenDimensions(AuxLossModule)`

```python
MixtureOfHiddenDimensions(
    dim: int,
    hidden_dim: int | None = None,
    expansion: float = 4.0,
    num_subdims: int = 16,
    active_ratio: float = 0.5,
    shared_ratio: float = 0.375,
    fusion: bool = True,
    aux_weight: float = 1e-2,
    bias: bool = False,
)
```

| 인자 | 기본값 | 의미 · 제약 |
|---|---|---|
| `dim` | — | 모델 차원 (입력 == 출력) |
| `hidden_dim` | `None`(=`expansion·dim`) | 중간 폭 `d_ff`. `num_subdims`로 나눠떨어져야 함 |
| `num_subdims` | `16` | sub-dim 그룹 수 `N` (논문 best) |
| `active_ratio` | `0.5` | 토큰당 활성 그룹 비율 `δ` (shared + specialized) |
| `shared_ratio` | `0.375` | 항상-on shared 그룹 비율 `φ`. `0 ≤ φ ≤ δ` (≈active의 3/4) |
| `fusion` | `True` | grouped fusion 적용 여부 |
| `aux_weight` | `1e-2` | load-balance 가중치 |
| `bias` | `False` | projection bias |

- 서브모듈: `w_up = Linear(dim, hidden_dim)`, `w_down = Linear(hidden_dim, dim)`,
  `centroids` 파라미터 `(num_subdims, dim)`, `fusion = _GroupedFusion(hidden_dim,
  groups=num_subdims)`(블록대각 linear).
- 내부 카운트: `n_shared = round(φ·N)`, `n_active = max(n_shared,
  round(δ·N))`, `n_specialized = n_active − n_shared`.

### 3.4 `MixtureOfLookupExperts(nn.Module)`

> aux loss를 내지 않으므로(dense routing) `AuxLossModule`이 아니라 순수
> `nn.Module`을 상속한다.

```python
MixtureOfLookupExperts(
    dim: int,
    num_experts: int = 4,
    routed_hidden: int | None = None,
    shared_hidden: int | None = None,
    expansion: float = 4.0,
    use_shared: bool = True,
    bias: bool = False,
)
```

| 인자 | 기본값 | 의미 |
|---|---|---|
| `dim` | — | 모델 차원 |
| `num_experts` | `4` | routed(lookup) expert 수 `N` |
| `routed_hidden` | `None`(=`expansion·dim`) | routed expert FFN 폭 |
| `shared_hidden` | `None`(=`expansion·dim`) | shared expert FFN 폭 |
| `expansion` | `4.0` | 폭 미지정 시 배수 |
| `use_shared` | `True` | shared(문맥) expert 포함 여부 |
| `bias` | `False` | FFN linear bias |

- `forward(h, e) -> (B, N, d)`: 학습용. `h` hidden 상태, `e` 토큰 임베딩.
- `build_lookup_table(embedding_weight) -> (V, N, d)`: 학습 후 1회 실행. LUT 사전계산.
- `forward_lookup(h, token_ids, lookup_table) -> (B, N, d)`: 추론용. routed
  expert FFN 연산 없이 `lookup_table[token_ids]`로 인덱싱.

### 3.5 `AttractorPatchNetwork(AuxLossModule)`

```python
AttractorPatchNetwork(
    dim: int,
    num_patches: int = 256,
    rank: int = 32,
    top_k: int = 4,
    tau: float = 0.07,
    gamma: float = 1.0,
    aux_weight: float = 1e-2,
)
```

| 인자 | 기본값 | 의미 · 제약 |
|---|---|---|
| `dim` | — | 토큰 차원 `d` (입력 == 출력) |
| `num_patches` | `256` | 프로토타입/패치 수 `K` |
| `rank` | `32` | compact-code 차원 `r` (`r ≪ d`) |
| `top_k` | `4` | 토큰당 활성 패치 수 `k`. `k ≤ K` |
| `tau` | `0.07` | 라우팅 온도 `τ` |
| `gamma` | `1.0` | 잔차 스케일 `γ` |
| `aux_weight` | `1e-2` | load-balance 가중치 |

- 파라미터: `prototypes (K, d)`, `code_proj V (d, r)`, `gate_a/gate_b (K, r)`,
  `decoder U (K, d, r)`, `norm = LayerNorm(dim)`.
- `forward(h) -> (B, T, d)`: cosine top-k 라우팅 → 공유 코드 → 저랭크 잔차 합.

---

## 4. 사용 예제

> 본 프로젝트는 **poetry**로 관리한다. 아래 예제는 모두
> `poetry run python`으로 검증됐다(plain `python` 금지).
> import: `from ypsilon_torch.blocks.mixture import (...)`,
> `from ypsilon_torch.blocks import AuxLossModule`.

### 4.1 MixtureOfDepths — FFN 래핑

```python
import torch
from torch import nn
from ypsilon_torch.blocks.mixture import MixtureOfDepths

ffn = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))
mod = MixtureOfDepths(block=ffn, dim=64, capacity_ratio=0.125).cuda()

x = torch.randn(2, 128, 64, device="cuda")
y = mod(x)                              # (2, 128, 64) — 12.5% 토큰만 ffn 통과

# 자기회귀 추론용 인과 라우팅 마스크
mod.eval()
route = mod.predict_route(x)            # (2, 128) bool
```

### 4.2 MixtureOfDepthsAndExperts — integrated / staged

```python
import torch
from ypsilon_torch.blocks.mixture import MixtureOfDepthsAndExperts

x = torch.randn(2, 128, 64, device="cuda")

# 논문 권장 integrated: experts + no-op 단일 top-1 라우터
mode = MixtureOfDepthsAndExperts(
    dim=64, num_experts=4, variant="integrated",
).cuda()
y = mode(x)                             # (2, 128, 64)

# staged: MoD 선별 후 별도 MoE
mode_s = MixtureOfDepthsAndExperts(
    dim=64, num_experts=4, variant="staged", capacity_ratio=0.25,
).cuda()
y_s = mode_s(x)                         # (2, 128, 64)
```

### 4.3 MixtureOfHiddenDimensions — FFN 대체

```python
import torch
from ypsilon_torch.blocks.mixture import MixtureOfHiddenDimensions

mohd = MixtureOfHiddenDimensions(
    dim=64, hidden_dim=256, num_subdims=16,
    active_ratio=0.5, shared_ratio=0.375,
).cuda()
x = torch.randn(2, 128, 64, device="cuda")
y = mohd(x)                             # (2, 128, 64)
```

### 4.4 MixtureOfLookupExperts — 학습 후 LUT 추론

```python
import torch
from torch import nn
from ypsilon_torch.blocks.mixture import MixtureOfLookupExperts

dim, vocab = 64, 100
emb = nn.Embedding(vocab, dim).cuda()
mole = MixtureOfLookupExperts(dim=dim, num_experts=4).cuda()

# --- 학습: 라우터는 h, routed expert는 임베딩 e ---
ids = torch.randint(0, vocab, (2, 16), device="cuda")
h = torch.randn(2, 16, dim, device="cuda")
e = emb(ids)
y = mole(h, e)                          # (2, 16, 64)

# --- 추론: LUT 사전계산 후 expert FFN 없이 인덱싱 ---
mole.eval()
with torch.no_grad():
    lut = mole.build_lookup_table(emb.weight)   # (vocab, N, dim)
    y_lut = mole.forward_lookup(h, ids, lut)    # (2, 16, 64)
    # 근사 아님 — 학습 경로와 정확히 동일
    assert torch.allclose(mole(h, emb(ids)), y_lut, atol=1e-5)
```

### 4.5 AttractorPatchNetwork — 잔차 블록

```python
import torch
from ypsilon_torch.blocks.mixture import AttractorPatchNetwork

apn = AttractorPatchNetwork(
    dim=64, num_patches=256, rank=32, top_k=4, tau=0.07,
).cuda()
h = torch.randn(2, 16, 64, device="cuda")
y = apn(h)                              # (2, 16, 64) — h + γ·Σ w_i Δ_i
```

### 4.6 auxloss를 학습 루프에 통합 (mixture + bilinear 동시 수집)

mixture 블록과 bilinear 블록 모두 `AuxLossModule`의 `reg_loss` 메커니즘을 쓰므로,
`collect` 한 번이 둘을 함께 모은다.

```python
import torch
from torch import nn
from ypsilon_torch.blocks import AuxLossModule
from ypsilon_torch.blocks.mixture import MixtureOfHiddenDimensions
from ypsilon_torch.blocks.bilinear_mlp import (
    AsymmetricSpatialChannelFactorizedBilinearMLP,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mohd = MixtureOfHiddenDimensions(dim=64, hidden_dim=256)
        self.afbo = AsymmetricSpatialChannelFactorizedBilinearMLP(
            dim=64, hidden_dim=256, num_groups=8, aux_loss_weight=0.1,
        )
    def forward(self, x):
        return self.afbo(self.mohd(x))

model = Model().cuda().train()

for batch, target in loader:
    logits = model(batch)
    primary = criterion(logits, target)

    # mixture + bilinear 블록의 aux loss를 한 번에 합산
    aux = AuxLossModule.reg_loss.collect(model)

    loss = primary + aux           # aux는 이미 각 aux_weight가 곱해진 값
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 다음 스텝을 위해 모든 모듈의 _reg_loss를 0으로 초기화
    AuxLossModule.reg_loss.reset(model, 0.0)
    # model 자신이 AuxLossModule이면 model.reg_loss = 0.0 도 가능
```

> `model`이 `AuxLossModule` 인스턴스일 필요는 없다. `collect` / `reset`은
> `model.modules()`를 순회하며 `_reg_loss`를 가진 모든 서브모듈을 찾는다.
> 단, `model.reg_loss = 0.0` **세터 문법**은 `model` 자체가 `AuxLossModule`을
> 상속할 때만 동작한다(위 예제처럼 plain `nn.Module`을 상속한 경우엔
> `AuxLossModule.reg_loss.reset(model, 0.0)`을 써야 전체 서브모듈이 초기화된다).

---

## 5. 하이퍼파라미터 가이드

### MoD: `capacity_ratio`

| 값 | 효과 |
|---|---|
| `0.125` (기본) | 논문 권장 12.5%. 토큰당 평균 연산을 크게 절감. |
| `0.25` ~ `0.5` | 보수적. 정확도 손실 적지만 절감폭도 작음. |
| `1.0` | 모든 토큰 처리. MoD 비활성(일반 블록과 동일). |

`aux_weight`: predictor를 학습에 쓰려면 `> 0`(기본 `1.0`). top-k 학습만 하고
자기회귀 추론을 안 한다면 `0.0`으로 둬도 된다(predictor는 여전히 사용 가능).

### MoDE: `variant` / `num_experts`

| `variant` | 특징 |
|---|---|
| `"integrated"` (기본) | 논문 권장. 깊이·expert 선택이 단일 라우터 공유. no-op 슬롯이 MoD 우회 역할. |
| `"staged"` | MoD 선별 후 별도 MoE. `capacity_ratio`로 처리량 직접 제어. |

`num_experts`는 보통 `4` ~ `8`. `aux_weight`는 Switch 관례를 따라 `1e-2` 권장
(너무 크면 라우터가 균등 분배에만 매달려 전문화가 약해진다).

### MoHD: `num_subdims` / `active_ratio` / `shared_ratio`

| 인자 | 권장 | 비고 |
|---|---|---|
| `num_subdims` | `16` | 논문 best. `hidden_dim`을 나눠떨어지게. |
| `active_ratio` (δ) | `0.5` | 토큰당 활성 그룹 절반. |
| `shared_ratio` (φ) | `0.375` | active의 ≈3/4를 shared로. `0 ≤ φ ≤ δ`. |
| `fusion` | `True` | 끄면(`False`) grouped fusion 제거, 더 가벼움. |

`shared_ratio`를 `active_ratio`와 같게 두면 specialized 그룹이 0이 되어 라우팅이
사라진다(전 그룹 shared = dense FFN에 가까움).

### MoLE: `num_experts` / `use_shared`

| 인자 | 권장 | 비고 |
|---|---|---|
| `num_experts` | `4` ~ `N` | dense routing이라 expert 수가 LUT 크기 `V×N×d`를 직접 좌우. |
| `use_shared` | `True` | shared expert가 `h` 문맥 용량 담당. 끄면 순수 lookup. |

LUT 메모리: `V × num_experts × dim`. 큰 vocab + 많은 expert면 LUT가 커지므로
`num_experts`로 추론 메모리를 조율한다.

### APN: `num_patches` / `rank` / `top_k` / `tau`

| 인자 | 기본 | 비고 |
|---|---|---|
| `num_patches` (K) | `256` | 패치 수. ⚠ 검증 필요 기본값(§2.5 주의). |
| `rank` (r) | `32` | compact-code 차원. `r ≪ d`. ⚠ 검증 필요. |
| `top_k` (k) | `4` | 토큰당 활성 패치. 논문 범위 `[2, 8]`. |
| `tau` (τ) | `0.07` | 라우팅 온도. 작을수록 sharp. ⚠ 검증 필요. |
| `gamma` (γ) | `1.0` | 잔차 스케일. |

`aux_weight`는 `1e-2` 권장. 너무 작으면 소수 패치만 쓰이는 붕괴(collapse)가
일어날 수 있다.

---

## 6. 구현 노트

### 공유 `reg_loss` (bilinear과 동일 메커니즘, 검증됨)

mixture 블록의 `AuxLossModule`은 `BilinearMLPBase`와 같은 `torchutils.auxloss`
기반 `reg_loss`를 쓴다. 한 모델 안에 둘이 섞여 있어도
`AuxLossModule.reg_loss.collect(model)` 한 번이면 둘의 손실이 함께 모인다.
검증 결과(`§4.6` 구조): combined collect가 비영(non-zero) 스칼라를 반환하고,
`reset` 직후 `0.0`이 된다.

`_reg_loss`는 `register_buffer(..., persistent=False)`라 `.to(device/dtype)`는
따라가지만 `state_dict`엔 들어가지 않는다 — 순간적 aux 값이 체크포인트에
새지 않는다.

### `eval()` / `aux_weight==0`에서 aux loss 스킵

MoD/MoDE/MoHD/APN 모두 `self.training and self.aux_weight != 0.0`일 때만 aux
loss를 계산하고, 그 외에는 `self.reg_loss = 0.0`을 쓴다. 추론 overhead를
없애기 위함이다. `collect`/`reset`은 그래도 정상 동작한다(슬롯은 항상 살아 있음).

### MoD: expert-choice → load-balance loss 불필요

MoD는 토큰이 expert를 고르는 token-choice가 아니라, 블록이 top-`C` 토큰을 고르는
**expert-choice**다. 이 방향에서는 처리량이 정확히 `C`로 고정돼 부하가 구조적으로
균등하다 — 그래서 별도 load-balance loss가 없고, 대신 인과 추론을 위한
predictor BCE만 둔다.

top-`C` 인덱스는 `torch.sort`로 원래 순서로 되돌린 뒤 `gather` → `block` →
`scatter_add_` 한다. 정렬은 래핑 블록이 causal일 때 토큰 순서를 보존하기 위함.

### MoLE: LUT는 근사가 아니라 정확

routed expert가 임베딩 `e`만 입력받으므로 `FFN_j(Embedding(v))`는 토큰 id `v`의
순수 함수다. `build_lookup_table`이 vocab 전체에 대해 이를 그대로 평가해
저장하고, `forward_lookup`은 동일 FFN을 인덱싱으로 대체할 뿐이다. 따라서 학습
경로와 LUT 경로의 출력은 **수치적으로 동일**하다 — 검증에서 maxdiff `0.0`.

### MoHD: grouped fusion = Monarch의 경량 대체

원논문은 grouped fusion에 **Monarch 행렬**(`O(d/r)` 비용)을 쓴다. 본 구현은
같은 "값싼 구조적 재혼합" 역할을 **블록대각 grouped linear**
(`einsum("...gb,gbc->...gc", xg, weight)`)로 대체한다 — Monarch permutation
없이 그룹 내부만 섞는다. `fusion=False`로 완전히 끌 수 있다.

α 스케일 `(Σ g_i)·N_sub`는 게이트로 0이 된 그룹 때문에 줄어든 활성 크기를 보정해
down-proj 입력 통계를 dense FFN과 비슷하게 유지한다.

### APN: 기본 하이퍼파라미터 검증 권장

`K=256`, `r=32`, `τ=0.07`, `k∈[2,8]`는 소스 코드에서 자동 reader가 추출한
값으로, docstring에 **검증 플래그**가 명시돼 있다. 그대로도 동작하지만 원논문과
대조 검증한 뒤 운영에 쓰기를 권장한다(이 문서·소스 모두 동일하게 명시).

저랭크 잔차는 `einsum("btkdr,btkr->btkd", U_sel, phi)`로 패치별 `U_i φ_i`를
계산한다. `U_sel`은 `decoder[topi]`로 gather되며 shape `(B, T, k, d, r)`.

---

## 7. 참고문헌

- **Mixture-of-Depths (MoD / MoDE)**:
  Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro,
  "Mixture-of-Depths: Dynamically allocating compute in transformer-based
  language models" — <https://arxiv.org/abs/2404.02258> (MoDE는 §4).
- **Mixture-of-Hidden-Dimensions (MoHD)**:
  Chen et al., "Mixture of Hidden-Dimensions: Not All Hidden-States'
  Dimensions are Needed in Transformer" (ICML 2025)
  — <https://arxiv.org/abs/2412.05644>.
- **Mixture-of-Lookup-Experts (MoLE)**:
  Jie, Tang, Han et al., "Mixture of Lookup Experts" (ICML 2025 Oral)
  — <https://arxiv.org/abs/2503.15798>.
- **Attractor Patch Networks (APN)**:
  "Attractor Patch Networks" — <https://arxiv.org/abs/2602.06993>.
  (기본 하이퍼파라미터는 자동 추출값 — 원논문 대조 검증 권장.)
- **Switch Transformer** — load-balancing loss의 출처:
  Fedus, Zoph, Shazeer, "Switch Transformers: Scaling to Trillion Parameter
  Models with Simple and Efficient Sparsity"
  — <https://arxiv.org/abs/2101.03961>.
- **`torchutils` / `auxloss`** — `reg_loss` 메커니즘:
  <https://github.com/newsniper-org/torchutils>.
