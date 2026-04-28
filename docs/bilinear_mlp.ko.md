# Bilinear MLP 블록 (AFBO 스타일 비대칭 Spatial-Channel Factorization)

> English version: [bilinear_mlp.en.md](bilinear_mlp.en.md)

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

`ypsilon_torch.blocks.bilinear_mlp.AsymmetricSpatialChannelFactorizedBilinearMLP`는
**Vision Transformer의 FFN(Feed-Forward Network)** 을 그대로 대체할 수 있는
블록이다. 두 가지 최근 아이디어를 하나로 결합한다.

- **Bilinear MLPs** (arXiv:2410.08417) — `y = W_out ((W₁x) ⊙ (W₂x))` 형태의
  MLP. 두 선형 projection 사이에 **element-wise 비선형성이 없으므로** 전체
  레이어를 3차 텐서 `B_{ijk}`로 정확히 표현할 수 있고, weight만 보고
  해석(eigendecomposition, circuit 식별 등)할 수 있다.
- **AFBO / SCFBO** (ICLR 2025) — ViT-FFN의 각 linear projection을
  **Spatial Modeling (SM)** 과 **Channel Mapping (CM)** 으로 분해하고, 두
  branch의 CM을 서로 다른 구조적 희소(sparsity) 전략(GCCM, OCCM)으로
  **비대칭(asymmetric)** 하게 구성해 성능/복잡도 trade-off를 개선한다.

이 블록은 두 branch를 모두 선형으로 유지(Bilinear MLP 성질 보존)하면서,
각 branch를 `CM(SM(x))`로 분해(AFBO 성질 보존)해 둘의 장점을 동시에 취한다.

### 데이터 흐름

```
                 ┌──────────── Branch A ────────────┐
 x (B,N,C) ──┬──►│  SM_a  ─►  GCCM  ─►  a (B,N,H) │──┐
             │   └──────────────────────────────────┘  │
             │                                         ├─► a ⊙ b ─► Dropout ─► W_out ─► y (B,N,C_out)
             │   ┌──────────── Branch B ────────────┐  │
             └──►│  SM_b  ─►  OCCM  ─►  b (B,N,H) │──┘
                 └──────────────────────────────────┘
```

- `SM`: 토큰 시퀀스 축을 따라 작동하는 **채널별(depth-wise)** 연산
  (dwconv / avg pool / identity).
- `GCCM`: **Grouped Cross Channel Mapping** — 비중첩 그룹 분할 + 채널 셔플.
- `OCCM`: **Overlapped Cycle Channel Mapping** — 순환 중첩 그룹 분할.
- 두 branch 출력의 element-wise 곱이 bilinear core를 이루고, 그 위에 출력
  projection `W_out`이 놓인다.

---

## 2. 수학적 정의

입력 `x ∈ ℝ^{B×N×C}`에 대해,

### Bilinear MLP core

$$
a = W_1 x, \qquad b = W_2 x, \qquad y = W_{\text{out}} (a \odot b).
$$

element-wise 비선형성이 전혀 없으므로 전체 연산은 입력에 대한 **3차 텐서**
로 정확히 쓸 수 있다:

$$
y_i \;=\; \sum_{j,k} B_{ijk}\, x_j\, x_k,
\qquad
B_{ijk} \;=\; \sum_{h} (W_{\text{out}})_{ih}\,(W_1)_{hj}\,(W_2)_{hk}.
$$

이 구조가 Pearce et al.이 말하는 **weight-based interpretability** 의 근거다.

### SCFBO 분해

각 선형 branch를 **Spatial Modeling**과 **Channel Mapping**의 합성으로
분해한다:

$$
a \;=\; \mathrm{CM}_a\bigl(\mathrm{SM}_a(x)\bigr), \qquad
b \;=\; \mathrm{CM}_b\bigl(\mathrm{SM}_b(x)\bigr).
$$

- `SM`: 토큰 시퀀스 축(길이 `N`)을 따라 채널별로 작동. dwconv(`groups=C`),
  `AvgPool1d`, 또는 identity.
- `CM`: 채널 축(`C → H`)을 따르는 **구조적 희소 linear** (아래 GCCM/OCCM).

### 비대칭: GCCM vs OCCM

두 branch의 CM을 **서로 다른** 구조적 희소 전략으로 구성한다. 그룹 수 `G`
가 주어졌을 때:

**GCCM — Grouped Cross Channel Mapping**

- 입력 채널을 `G`개의 비중첩 그룹(각 크기 `C/G`)으로 분할.
- 각 그룹에 독립적인 linear `W_g ∈ ℝ^{(C/G)×(H/G)}`를 적용.
- 이후 **cyclic channel shuffle**(ShuffleNet과 동일)로 출력 채널을 재배열해
  다음 layer (또는 bilinear 쌍의 반대편 branch)에서 그룹 간 정보가 섞이게
  한다.
- 파라미터 수: `G · (C/G) · (H/G) = C · H / G` (full linear 대비 `G`배 절감).

**OCCM — Overlapped Cycle Channel Mapping**

- 각 출력 그룹이 입력 채널을 **순환 중첩**으로 읽는다. 그룹 `g`는 채널
  `[g·(C/G) − s : g·(C/G) + (C/G) + s] mod C` (overlap `s`) 를 본다.
- 각 그룹별 linear weight: `W_g ∈ ℝ^{(C/G + 2s)×(H/G)}`.
- 파라미터 수: `G · (C/G + 2s) · (H/G) = C · H / G + 2s · H`.
- `s = 0`이면 GCCM에서 shuffle만 제거한 pure grouped linear과 동일.

**왜 비대칭인가?** GCCM의 channel shuffle은 "출력 순서를 교차"시켜 **광역**
그룹 상호작용을 주고, OCCM의 cyclic overlap은 **국소** 이웃 그룹과 weight
레벨에서 직접 연결된다. 이 두 전략은 상보적이어서, bilinear 곱
`a ⊙ b`에서 서로 다른 종류의 cross-group 정보를 교환하는 효과를 낸다.
AFBO 논문이 주장하는 성능/복잡도 trade-off의 핵심.

### Bilinearity identity (검증용)

모든 bias를 0으로 두면 이 블록은 **입력에 대해 정확히 2차 동차(homogeneous
of degree 2)** 이다:

$$
\text{forward}(\alpha x) \;=\; \alpha^2 \,\text{forward}(x).
$$

실측 확인: `α = 2`일 때 출력 비율이 `≈ 4.0`으로 나와야 한다 (§6 구현 노트
참조).

---

## 3. 아키텍처 및 API 레퍼런스

### 3.1 `BilinearMLPBase(torch.nn.Module, abc.ABC)`

모든 bilinear-MLP 블록의 추상 베이스 클래스.

**생성자**

```python
BilinearMLPBase(
    dim: int,
    hidden_dim: int | None = None,   # 기본값: int(dim * expansion)
    out_dim: int | None = None,      # 기본값: dim
    expansion: float = 4.0,
    dropout: float = 0.0,
    bias: bool = True,
)
```

| 인자 | 의미 |
|---|---|
| `dim` | 입력 채널 수 `C_in` |
| `hidden_dim` | bilinear 곱이 일어나는 차원 `H`. `None`이면 `dim * expansion`. |
| `out_dim` | 출력 채널 수 `C_out`. `None`이면 `dim`. |
| `expansion` | `hidden_dim`이 `None`일 때만 사용 |
| `dropout` | bilinear 곱 이후 `W_out` 직전 dropout 확률 |
| `bias` | `out_proj`의 bias 유무 |

**필수 오버라이드 메서드** (서브클래스에서 구현)

```python
@abc.abstractmethod
def branch_a(self, x: Tensor) -> Tensor: ...   # (B,N,dim) → (B,N,hidden_dim)
@abc.abstractmethod
def branch_b(self, x: Tensor) -> Tensor: ...   # (B,N,dim) → (B,N,hidden_dim)
```

**`forward`**

```python
a = self.branch_a(x)
b = self.branch_b(x)
self._compute_aux_loss(a, b)        # reg_loss 슬롯 업데이트 (서브클래스 훅)
h = self.dropout(a * b)
return self.out_proj(h)
```

**`reg_loss` property (torchutils.auxloss, 필수 `>= 0.1.1`)**

- 게터: 현재 모듈의 `_reg_loss` 스칼라 tensor 반환.
- 세터: `self.reg_loss = loss` (Tensor 또는 `float`). 내부적으로 dtype/shape
  정규화.
- 수집기 (**static-style 호출**): `BilinearMLPBase.reg_loss.collect(model)` —
  `model.modules()` 를 순회하며 모든 `_reg_loss`를 `torch.sum`으로 집계.
- 리셋터: `BilinearMLPBase.reg_loss.reset(model, 0.0)` 또는
  `model.reg_loss = 0.0`로 모든 sub-module의 `_reg_loss`를 스칼라 값으로
  초기화.

### 3.2 `AsymmetricSpatialChannelFactorizedBilinearMLP(BilinearMLPBase)`

AFBO 스타일 구체 블록.

**생성자**

```python
AsymmetricSpatialChannelFactorizedBilinearMLP(
    dim: int,
    hidden_dim: int | None = None,
    out_dim: int | None = None,
    expansion: float = 4.0,
    dropout: float = 0.0,
    bias: bool = True,
    sm_kind: Literal["dwconv", "pool", "none"] = "dwconv",
    sm_kernel_size: int = 3,
    num_groups: int = 8,
    occm_overlap: int = 1,
    aux_loss_weight: float = 0.0,
)
```

| 인자 | 기본값 | 의미 · 제약 |
|---|---|---|
| `sm_kind` | `"dwconv"` | spatial modeling 종류. `"dwconv"` / `"pool"` / `"none"` |
| `sm_kernel_size` | `3` | `≥ 1` |
| `num_groups` | `8` | `G`. `dim % G == 0` 과 `hidden_dim % G == 0` 필요 |
| `occm_overlap` | `1` | `s`. `0 ≤ 2s ≤ dim / G` |
| `aux_loss_weight` | `0.0` | branch-decorrelation regularization 가중치 |

**서브모듈**

- `sm_a`, `sm_b`: `SpatialModeling` 2개 (각 branch 독립).
- `cm_a`: `GroupedCrossChannelMapping(dim, hidden_dim, num_groups)`
- `cm_b`: `OverlappedCycleChannelMapping(dim, hidden_dim, num_groups, occm_overlap)`

### 3.3 `SpatialModeling(nn.Module)`

```python
SpatialModeling(
    dim: int,
    kind: Literal["dwconv", "pool", "none"] = "dwconv",
    kernel_size: int = 3,
    bias: bool = False,
)
```

- 입력/출력: `(B, N, C)`.
- 내부에서 `(B, N, C) → (B, C, N)`로 전치한 뒤 Conv1d / AvgPool1d 적용,
  다시 `(B, N, C)`로 전치.
- padding은 `kernel_size // 2` (same padding).

### 3.4 `GroupedCrossChannelMapping(nn.Module)`

```python
GroupedCrossChannelMapping(
    in_channels: int,
    out_channels: int,
    num_groups: int,
    bias: bool = True,
)
```

- 가중치 shape: `(G, in_channels/G, out_channels/G)`.
- 순전파: `einsum("...gi,gio->...go", x_g, W)` 후 cyclic channel shuffle
  (전치 + reshape).
- `nn.Linear(in_channels, out_channels)` 의 drop-in 대체 (마지막 축 기준).

### 3.5 `OverlappedCycleChannelMapping(nn.Module)`

```python
OverlappedCycleChannelMapping(
    in_channels: int,
    out_channels: int,
    num_groups: int,
    overlap: int = 1,
    bias: bool = True,
)
```

- 가중치 shape: `(G, in_channels/G + 2·overlap, out_channels/G)`.
- `gather_idx`를 non-persistent buffer로 미리 계산: shape `(G, window)`에
  각 그룹이 읽을 입력 채널 인덱스를 순환적으로 저장.
- 순전파: `x.index_select(-1, gather_idx.flatten())` → reshape →
  `einsum("...gw,gwo->...go", x_g, W)`.

---

## 4. 사용 예제

### 4.1 최소 예제

```python
import torch
from ypsilon_torch.blocks import AsymmetricSpatialChannelFactorizedBilinearMLP

block = AsymmetricSpatialChannelFactorizedBilinearMLP(
    dim=64, hidden_dim=256, num_groups=8,
).cuda()
x = torch.randn(2, 16, 64, device="cuda")
y = block(x)                         # (2, 16, 64)
```

### 4.2 ViT FFN 대체

```python
import torch
from torch import nn
from ypsilon_torch.blocks import AsymmetricSpatialChannelFactorizedBilinearMLP

class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        # ---- 기존 FFN 대신 Bilinear MLP 사용 ----
        self.ffn = AsymmetricSpatialChannelFactorizedBilinearMLP(
            dim=dim,
            expansion=mlp_ratio,
            num_groups=8,
            occm_overlap=1,
            sm_kind="dwconv",
        )

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x
```

### 4.3 auxloss를 학습 루프에 통합

`aux_loss_weight > 0.0`으로 생성하면, `forward` 시 두 branch의 cosine-square
유사도가 `reg_loss` 슬롯에 기록된다.

```python
from ypsilon_torch.blocks import BilinearMLPBase

model = MyViT(...).cuda()

for batch, target in loader:
    logits = model(batch)
    primary = criterion(logits, target)

    # 모델 전체의 bilinear MLP 블록들의 aux loss를 합산
    aux = BilinearMLPBase.reg_loss.collect(model)

    loss = primary + aux           # aux는 이미 aux_loss_weight가 곱해진 값
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 다음 스텝을 위해 모든 모듈의 _reg_loss를 0으로 초기화
    model.reg_loss = 0.0
    # 또는: BilinearMLPBase.reg_loss.reset(model, 0.0)
```

> `model`이 `BilinearMLPBase` 인스턴스일 필요는 없다. `reg_loss.collect` /
> `.reset`은 `model.modules()`를 순회하며 `_reg_loss` 속성을 가진 모든
> 서브모듈을 찾는다.

### 4.4 3차 텐서 복원 (interpretability)

```python
import torch

# block.out_proj: Linear(H, C_out),  H = hidden_dim
# branch_a, branch_b: 선형이므로 Jacobian이 곧 weight 행렬

B = 1 # batch
x0 = torch.zeros(B, 1, block.dim)
W_out = block.out_proj.weight          # (C_out, H)
# 각 branch의 effective linear matrix (Jacobian ∂a/∂x, ∂b/∂x)
W1 = torch.autograd.functional.jacobian(
    lambda x: block.branch_a(x.unsqueeze(0).unsqueeze(0))[0, 0], x0[0, 0]
)
W2 = torch.autograd.functional.jacobian(
    lambda x: block.branch_b(x.unsqueeze(0).unsqueeze(0))[0, 0], x0[0, 0]
)
# y_i = Σ_{j,k} B_{ijk} x_j x_k,  B_{ijk} = Σ_h W_out[i,h] W1[h,j] W2[h,k]
B_tensor = torch.einsum("ih,hj,hk->ijk", W_out, W1, W2)
# B_tensor를 고유분해하면 Pearce et al. 방식의 interpretability 분석이 가능
```

---

## 5. 하이퍼파라미터 가이드

### `num_groups` (G)

| 값 | 효과 |
|---|---|
| `1` | GCCM/OCCM이 full linear가 된다 (비대칭 사라짐). 추천하지 않음. |
| `4` ~ `8` | 균형 잡힌 trade-off. 일반적 기본값. |
| `16`, `32` | 파라미터 `G`배 절감. 작은 모델, 경량 추론용. |

제약: `dim % G == 0`, `hidden_dim % G == 0`.

### `occm_overlap` (s)

| 값 | 효과 |
|---|---|
| `0` | OCCM이 pure grouped linear로 퇴화. shuffle 없는 GCCM과 유사. |
| `1` (기본) | 이웃 그룹 각각에서 1채널씩 leakage. 권장. |
| `≥ 2` | 더 강한 국소 cross-group 연결. `2s ≤ dim/G` 제약 유의. |

### `sm_kind` / `sm_kernel_size`

| `sm_kind` | 특징 |
|---|---|
| `"dwconv"` (기본) | AFBO 기본값. 학습 가능, 가장 표현력 좋음. |
| `"pool"` | 파라미터 0. 기준선(baseline)/경량용. |
| `"none"` | 공간 모델링 완전 제거. pure channel-mixing MLP에 가까움. |

`sm_kernel_size`는 일반적으로 `3`; `7`로 키우면 더 넓은 이웃을 본다.

### `aux_loss_weight`

| 값 | 의미 |
|---|---|
| `0.0` (기본) | aux loss 계산 자체를 생략(`training` 모드여도 0 기록). `reg_loss` 슬롯은 여전히 살아 있어 `collect`/`reset`이 정상 동작. |
| `0.01` ~ `0.1` | branch-decorrelation 약한 규제. 일반적 범위. |
| `> 0.1` | 강한 규제. 두 branch가 지나치게 복제되는 모델에서만 사용. |

주의: `eval()` 모드에서는 weight가 양수여도 aux loss 계산을 건너뛴다 (추론
overhead 방지).

---

## 6. 구현 노트

### `_reg_loss`를 non-persistent buffer로

```python
self.register_buffer("_reg_loss", torch.zeros(()), persistent=False)
```

- `model.to(device)` / `.to(dtype)`에 자동으로 따라간다.
- `persistent=False`이므로 `state_dict`에 포함되지 않는다 — 학습 상태
  (순간적 aux loss 값)가 체크포인트에 들어가는 일을 방지.

### `torchutils >= 0.1.1` 필수

v0.1.0의 `auxloss.setter`는 `property.setter`의 `type(self)(fget, fset, fdel,
doc)` 호출을 그대로 받지만, `auxloss.__init__`의 **위치 인자 순서는
`(fget, fcollect, fset, freset, doc)`** 이어서 사용자 setter가 `fcollect`로
잘못 저장된다(그리고 실제 `fset`은 `None`). 결과적으로 `self.reg_loss = x`
가 `AttributeError: property 'reg_loss' has no setter`를 던진다.

v0.1.1의 "fix setter bugs" 릴리스가 `auxloss(fget, fcollect, fset=fset,
freset, doc)`를 직접 재구성하도록 수정해 이 문제를 해결했다. 본 블록은
v0.1.1 이상이 필요하며, 이는 `pyproject.toml`의 `rev = "v0.1.1"`로 고정되어
있다.

### Channel shuffle = ShuffleNet의 그것

`GroupedCrossChannelMapping`의 마지막 단계

```python
y_g = y_g.transpose(-2, -1).contiguous().reshape(*lead, out_channels)
```

는 ShuffleNet v1/v2의 channel shuffle과 동일한 permutation이다. 그룹별
출력 채널을 인터리브해 "다음 layer가 다른 그룹의 정보를 보게" 만든다.
이 블록에서는 "다음 layer" 역할을 bilinear 곱의 반대편 branch가 수행한다.

### Bilinearity 검증 (α² 성질)

구현 상 모든 bias를 0으로 두면 `forward(αx) = α² forward(x)` 가 해석적으로
성립한다. GPU 검증 결과 `α = 2`에서 `|y(2x)/y(x)|` 평균이 `4.0000`으로
확인된다. 이 성질이 깨지면 어딘가에 비선형성이 섞인 것이다.

### 파라미터 절감 실측

`dim=64, hidden_dim=256, G=8, overlap=1`:

| 모듈 | 파라미터 수 |
|---|---|
| full `nn.Linear(64, 256)` + bias | 16 640 |
| GCCM | 2 304 (7.2× 절감) |
| OCCM | 2 816 (5.9× 절감) |

### CUDA 성능

- einsum 경로(`"...gi,gio->...go"` 등)는 PyTorch 2.10의 `torch.einsum`이
  cuBLAS grouped GEMM으로 자동 fuse되어 추가 contiguity 비용이 실질적으로
  없다.
- `OCCM`의 `index_select`는 gather가 가벼워 forward/backward에서 병목이
  되지 않는다.

---

## 7. 참고문헌

- **AFBO (ICLR 2025)**:
  "Asymmetric Factorized Bilinear Operation for Vision Transformer"
  — <https://openreview.net/forum?id=MJyqwBVgMs>,
  <https://iclr.cc/virtual/2025/poster/29941>
- **Bilinear MLPs (arXiv 2024)**:
  "Bilinear MLPs enable weight-based mechanistic interpretability"
  — <https://arxiv.org/abs/2410.08417>
- **ShuffleNet (CVPR 2018, v2: ECCV 2018)** — channel shuffle의 원조 구현.
- **`torchutils` / `auxloss` 사용 패턴**:
  <https://github.com/newsniper-org/torchutils>,
  <https://github.com/Honey-Be/framesmoothie> (실사용 예제).
