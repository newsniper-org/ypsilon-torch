# Feed-Forward(FFN) 대체 블록

> English version: [feedforward.en.md](feedforward.en.md)

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

`ypsilon_torch.blocks.feedforward`는 **Transformer/Vision Transformer의
FFN(Feed-Forward Network)** 을 그대로 대체할 수 있는 두 가지 블록을 제공한다.
둘 다 입력과 출력 차원이 같은 `(B, N, d_model)` drop-in 모듈이다.

- **`HourglassFFN`** (arXiv:2602.06471 / 2510.01796) — 표준 FFN이
  `d_model → d_ff`(넓힘) `→ d_model`로 한 번 부풀렸다 줄이는 것과 달리,
  bottleneck `d_h < d_model`로 **좁힌** 뒤 `K`개의 residual sub-MLP를 쌓는다.
  너비(width)에 쓰던 파라미터를 깊이(depth)에 재배분하는 "모래시계(hourglass)"
  형태다. 각 sub-MLP는 SwiGLU형이다.
- **`MultiHeadFFN`** (arXiv:2512.06989 "Flash Multi-Head FFN") — SwiGLU FFN을
  **head별로 동적 가중되는 병렬 sub-network** 들의 집합으로 재해석한 블록.
  본 구현은 원논문의 **하드웨어 독립 순수 PyTorch 재구현**이다.

### `MultiHeadFFN`의 하드웨어 lock-in에 대하여

원논문 arXiv:2512.06989은 Triton 커널과, 더 나아가 NVIDIA Hopper/H100 전용
ThunderKittens(TMA / WGMMA / warp-group specialization) 기반 FlashAttention
스타일 fused 커널에 의존한다. 그러나 그 커널은 **순수하게 I/O를 최적화하는
장치일 뿐**이다. 즉 전체 `L × d_ff` 중간 텐서를 메모리에 실체화하지 않을 뿐,
수치적으로는 단순한 순수 연산 정식화와 **완전히 동일**하다. 따라서 하드웨어
lock-in(Hopper 전용 기능)은 **아키텍처가 아니라 속도/메모리 계층에만**
존재한다.

본 모듈은 논문의 아키텍처(Eq. 10–14)를 커널·custom op·가속기 종속성 없이 순수
PyTorch로 재현하므로, CPU·소비자용 GPU·데이터센터 GPU에서 모두 동일하게
동작한다. 선택적 `memory_efficient` 경로는 sub-network에 대해 Python 루프로
누적해 `L × d_ff` 텐서 생성을 피함으로써 메모리 이득의 대부분을 회복한다(속도
이득은 fused 커널이 필요하며 여기서는 재현하지 않는다).

### 데이터 흐름

**HourglassFFN** — bottleneck으로 좁히는 residual sub-MLP 스택:

```
 x (B,N,d_model)
   │
   ├──────────────────────────────┐  (residual)
   ▼                              │
 norm ─► W_d1 ─► SiLU ─┐          │
   └──► W_d2 ──────────┤⊙─► W_u ──┤+──►  h₁
                       (d_h)      │
   ┆        ... × K (depth) ...   ┆
   ▼                              │
 h_K (B,N,d_model)
```

**MultiHeadFFN** — head별 동적 가중 병렬 sub-network:

```
 x (B,N,d_model)
   │
 W_in ─► reshape ─► Q (B,N,H,d_h)
   │
   ├─► 게이트:  R^h = normalize(σ(Q^h·W_g^h))   (B,N,H,E)
   │
   └─► head h, sub-network e=1..E:
          A_e = SiLU(Q^h·K_eᵀ) ⊙ (Q^h·U_eᵀ)        (d_e)
          S^h = Σ_e R^h_e · (A_e · V_e)             (d_h)
                              │
            concat_H(S) ─► W_out ─►  O (B,N,d_model)
```

---

## 2. 수학적 정의

### 2.1 HourglassFFN

입력 `x ∈ ℝ^{B×N×d_model}`에 대해, `h_0 = x`로 두고 `K`개의 residual
sub-MLP를 거친다:

$$
h_{i+1} \;=\; h_i + W_u^{(i)}\!\Big(
   \operatorname{SiLU}\!\big(W_{d1}^{(i)} \bar h_i\big)
   \;\odot\; \big(W_{d2}^{(i)} \bar h_i\big)\Big),
\qquad
\bar h_i = \operatorname{norm}(h_i).
$$

출력은 `h_K`이다. 각 sub-MLP는 down-projection 두 개
`W_{d1}, W_{d2} \in \mathbb{R}^{d_h \times d_{model}}` 와 up-projection
`W_u \in \mathbb{R}^{d_{model} \times d_h}` 로 이루어진 **SwiGLU** 게이트이며,
중간 차원은 bottleneck `d_h = \mathrm{round}(d_{model}\cdot r) < d_{model}`
이다 (`r`은 `bottleneck_ratio`).

표준 FFN이 `d_ff > d_model`로 넓히는 것과 달리, 여기서는 `d_h < d_model`로
좁히고 그렇게 아낀 파라미터를 깊이 `K`에 투자한다.

### 2.2 MultiHeadFFN

입력 `X ∈ ℝ^{B×N×d_model}`에 대해, 먼저 `W_in`으로 사상한 뒤 `H`개 head로
재배열한다 (`d_h = d_{model}/H`):

$$
Q = \mathrm{reshape}\big(X W_{in},\ (B, N, H, d_h)\big).
$$

head별 동적 게이트 `R^h \in \mathbb{R}^{B \times N \times E}` 는 `E`개
sub-network 가중치이며, 시그모이드 후 합으로 정규화된다:

$$
R^h = \mathrm{normalize}\big(\sigma(Q^h W_g^h)\big),
\qquad
R^h_e = \frac{\sigma(Q^h W_g^h)_e}{\sum_{e'} \sigma(Q^h W_g^h)_{e'} + \varepsilon}.
$$

각 head 출력 `S^h`는 `E`개 sub-network의 SwiGLU 출력을 동적 가중 합한 것이다:

$$
S^h \;=\; \sum_{e=1}^{E} R^h_{e}\;
   \Big(\operatorname{SiLU}\!\big(Q^h K_e^{h\top}\big)
   \odot \big(Q^h U_e^{h\top}\big)\Big) V_e^h,
$$

여기서 각 `(head, sub-network)`마다
`K_e^h, U_e^h, V_e^h \in \mathbb{R}^{d_e \times d_h}` 이다. 마지막으로 head를
concat한 뒤 출력 사상한다:

$$
O = \mathrm{concat}_H(S)\, W_{out}.
$$

`Q^h K_e^{h\top}` 와 `Q^h U_e^{h\top}` 는 sub-network 내부 차원 `d_e`(기본
`round((8/3)·d_h)`)로 올리는 SwiGLU 게이트의 두 분기, `V_e^h`는 다시 `d_h`로
내리는 projection이다.

---

## 3. 아키텍처 및 API 레퍼런스

### 3.1 `HourglassFFN(nn.Module)`

**생성자**

```python
HourglassFFN(
    dim: int,
    bottleneck_ratio: float = 0.5,
    depth: int = 4,
    norm: Literal["layernorm", "rmsnorm", "none"] = "layernorm",
    bias: bool = False,
    hidden_dim: int | None = None,
)
```

| 인자 | 기본값 | 의미 · 제약 |
|---|---|---|
| `dim` | — | 모델 차원 `d_model` (입력 == 출력). |
| `bottleneck_ratio` | `0.5` | `d_h / d_model`. 권장 `0.4`–`0.6` (hourglass 영역). `≥ 1`도 허용되나 취지에 어긋남. |
| `depth` | `4` | 쌓는 sub-MLP 개수 `K`. 권장 `2`–`4`. `≥ 1` 필요. |
| `norm` | `"layernorm"` | sub-MLP별 pre-normalization. `"layernorm"` / `"rmsnorm"` / `"none"`. |
| `bias` | `False` | 선형 레이어의 bias 유무. |
| `hidden_dim` | `None` | bottleneck 크기 명시 지정. 주어지면 `bottleneck_ratio`를 무시. |

**서브모듈**

- `blocks`: `_HourglassSubMLP` `K`개를 담은 `nn.ModuleList`.
- 각 `_HourglassSubMLP`: `norm`, `w_d1`(`dim→hidden_dim`),
  `w_d2`(`dim→hidden_dim`), `w_u`(`hidden_dim→dim`).

**`forward`**

```python
h = x
for block in self.blocks:        # block(h) = w_u(SiLU(w_d1(norm(h))) * w_d2(norm(h)))
    h = h + block(h)
return h                          # (B, N, dim)
```

### 3.2 `MultiHeadFFN(nn.Module)`

**생성자**

```python
MultiHeadFFN(
    dim: int,
    num_heads: int = 8,
    num_subnetworks: int = 8,
    subnetwork_dim: int | None = None,
    bias: bool = False,
    eps: float = 1e-6,
    memory_efficient: bool = True,
)
```

| 인자 | 기본값 | 의미 · 제약 |
|---|---|---|
| `dim` | — | 모델 차원 `d_model` (입력 == 출력). `num_heads`로 나누어떨어져야 함. |
| `num_heads` | `8` | head 수 `H`. `d_h = dim / H`. |
| `num_subnetworks` | `8` | head별 병렬 sub-network 수 `E`. |
| `subnetwork_dim` | `None` | sub-network 너비 `d_e`. `None`이면 `round((8/3)·d_h)` (SwiGLU 확장). |
| `bias` | `False` | `W_in` / `W_out`의 bias 유무. |
| `eps` | `1e-6` | 게이트 정규화 안정화 항 `ε`. |
| `memory_efficient` | `True` | `True`면 `E`에 대한 Python 루프로 누적, `(B,N,H,E,d_e)` 중간 텐서 미생성. |

**파라미터**

- `w_in`, `w_out`: `nn.Linear(dim, dim)`.
- `w_gate`: `nn.Parameter` shape `(H, d_h, E)` — head별 게이트.
- `k`, `u`, `v`: 각 `nn.Parameter` shape `(H, E, d_e, d_h)` —
  `(head, sub-network)`별 K/U/V projection.

**`forward`** (`memory_efficient=True` 경로)

```python
q = self.w_in(x).reshape(B, N, H, dh)
p = torch.einsum("bnhd,hde->bnhe", q, self.w_gate)
r = torch.sigmoid(p)
r = r / (r.sum(dim=-1, keepdim=True) + self.eps)

s = torch.zeros_like(q)
for e in range(E):
    qk = torch.einsum("bnhd,hfd->bnhf", q, self.k[:, e])   # (B,N,H,d_e)
    qu = torch.einsum("bnhd,hfd->bnhf", q, self.u[:, e])
    a = F.silu(qk) * qu
    contrib = torch.einsum("bnhf,hfd->bnhd", a, self.v[:, e])
    s = s + r[..., e : e + 1] * contrib

return self.w_out(s.reshape(B, N, self.dim))
```

---

## 4. 사용 예제

> 이 프로젝트는 Poetry로 관리된다. 아래 스니펫은 모두
> `poetry run python`으로 실행/검증하라 (plain `python` 금지).

### 4.1 최소 예제

```python
import torch
from ypsilon_torch.blocks.feedforward import HourglassFFN, MultiHeadFFN

x = torch.randn(2, 16, 256)

hg = HourglassFFN(dim=256, bottleneck_ratio=0.5, depth=4)
y1 = hg(x)                           # (2, 16, 256)

mhf = MultiHeadFFN(dim=256, num_heads=8, num_subnetworks=8)
y2 = mhf(x)                          # (2, 16, 256)
```

### 4.2 ViT/Transformer FFN 대체 (drop-in)

```python
import torch
from torch import nn
from ypsilon_torch.blocks.feedforward import HourglassFFN, MultiHeadFFN

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn: str = "hourglass"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        # ---- 표준 FFN 자리에 drop-in ----
        if ffn == "hourglass":
            self.ffn = HourglassFFN(dim=dim, bottleneck_ratio=0.5, depth=4)
        else:
            self.ffn = MultiHeadFFN(dim=dim, num_heads=num_heads, num_subnetworks=8)

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x
```

`HourglassFFN`은 내부에 자체 residual·norm을 두므로 위처럼 `x + ffn(norm(x))`
형태의 외부 residual과 함께 써도 무방하다 (이중 residual은 학습을 해치지
않는다). 외부 norm/residual을 생략하고 `x = self.ffn(x)`만으로도 동작한다.

### 4.3 `memory_efficient` 경로 등가성 확인

```python
import torch
from ypsilon_torch.blocks.feedforward import MultiHeadFFN

x = torch.randn(2, 32, 256)

m = MultiHeadFFN(dim=256, num_heads=8, num_subnetworks=8, memory_efficient=True)
m.eval()
with torch.no_grad():
    y_loop = m(x)
    m.memory_efficient = False
    y_batched = m(x)

print((y_loop - y_batched).abs().max())   # ~1e-9 (수치적으로 동일)
```

---

## 5. 하이퍼파라미터 가이드

### HourglassFFN

#### `bottleneck_ratio` (r = d_h / d_model)

| 값 | 효과 |
|---|---|
| `0.4`–`0.6` (권장) | hourglass 영역. 너비를 줄여 아낀 파라미터를 깊이에 투자. |
| `≈ 1.0` | bottleneck 효과 사라짐. 깊은 residual MLP에 가까움. |
| `≥ 1.0` | 허용되나 hourglass 취지에 어긋남(넓힘). |

#### `depth` (K)

| 값 | 효과 |
|---|---|
| `1` | 단일 SwiGLU bottleneck sub-MLP. |
| `2`–`4` (권장) | 깊이로 표현력 확보. 표준 FFN 대체의 일반적 범위. |
| `> 4` | 더 깊지만 잔차 누적·연산량 증가. |

#### `norm`

| 값 | 특징 |
|---|---|
| `"layernorm"` (기본) | 안정적 기본값. |
| `"rmsnorm"` | LM 계열에서 흔함. bias·평균 빼기 없음, 약간 가벼움. |
| `"none"` | pre-norm 제거. 외부에서 norm을 주는 구조에서만 권장. |

### MultiHeadFFN

#### `num_heads` (H) / `num_subnetworks` (E)

| 인자 | 효과 |
|---|---|
| `num_heads` | `d_h = dim/H`를 결정. `dim % H == 0` 필수. H가 크면 head별 차원이 작아짐. |
| `num_subnetworks` | head별 병렬 전문가 수. 크면 표현력·파라미터·연산량 증가. `8` 부근이 일반적. |

#### `subnetwork_dim` (d_e)

| 값 | 효과 |
|---|---|
| `None` (기본) | `round((8/3)·d_h)` — 논문의 SwiGLU 확장 비율. |
| 명시 지정 | sub-network 내부 너비를 직접 조절. 작게 두면 경량화. |

#### `memory_efficient`

| 값 | 의미 |
|---|---|
| `True` (기본) | `E` 루프 누적. `(B,N,H,E,d_e)` 미생성 → 메모리 절약. 큰 `E`/긴 시퀀스에 유리. |
| `False` | batched einsum. 작은 `E`에서 약간 빠를 수 있으나 메모리 더 사용. 수치는 동일. |

---

## 6. 구현 노트

### 하드웨어 lock-in 제거의 의미 (MultiHeadFFN)

원논문 arXiv:2512.06989의 성능 기여는 Hopper/H100 전용
ThunderKittens(TMA/WGMMA/warp-group) fused 커널에서 나오지만, 그 커널은
`L × d_ff` 중간을 실체화하지 않는 **I/O-aware 최적화**일 뿐이며 결과는 순수
정식화와 수치적으로 동일하다. 따라서 본 재구현은 **아키텍처(Eq. 10–14)를
온전히 보존**하면서 커널 종속성만 제거했다. 그 결과 CPU·소비자용
GPU·데이터센터 GPU에서 동일하게 돌아가며, 잃는 것은 fused 커널의 속도
이득뿐이다(메모리 이득의 대부분은 `memory_efficient` 경로로 회복).

### `memory_efficient` vs `batched` 등가성

두 경로는 동일한 수식을 계산한다.

```python
# memory_efficient=True:  E에 대한 Python 루프로 누적, (B,N,H,E,d_e) 미생성
for e in range(E):
    a = F.silu(q @ k[:, e].T) * (q @ u[:, e].T)
    s += r[..., e:e+1] * (a @ v[:, e])

# memory_efficient=False:  (B,N,H,E,d_e) 한 번에 만들고 E축으로 합
a = F.silu(einsum(q, k)) * einsum(q, u)         # (B,N,H,E,d_e)
s = (r.unsqueeze(-1) * einsum(a, v)).sum(dim=3) # (B,N,H,d_h)
```

부동소수 합산 순서 차이만 있을 뿐이어서 출력은 수치적으로 동일하다(검증됨,
`max|Δ| ~ 1e-9`). 학습/추론 어느 쪽을 골라도 모델 동작은 같으며, 선택은
오직 메모리/속도 trade-off 문제다.

### HourglassFFN의 파라미터 효율

각 sub-MLP는 `W_{d1}, W_{d2} (d_model→d_h)` 와 `W_u (d_h→d_model)`,
즉 블록당 약 `3 · d_h · d_model` 파라미터를 쓴다. 따라서 `K`개 스택 전체는
대략

$$
K \cdot 3 \cdot d_h \cdot d_{model}
$$

이다 (bias 제외). 표준 FFN(`2 · d_ff · d_model`, 보통 `d_ff = 4·d_model`)과
비교하면, bottleneck `d_h < d_model` 덕분에 같은 파라미터 예산을 **너비
대신 깊이**로 옮겨 쓸 수 있다. 예: `d_model=256, r=0.5 (d_h=128), K=4`이면
블록당 `≈ 3·128·256 = 98304`, 4 스택 `≈ 393216` — `d_ff=1024` 표준
FFN(`≈ 2·1024·256 = 524288`)보다 작으면서 깊이는 4배다.

### init

`MultiHeadFFN.reset_parameters`는 구조화 파라미터 `k`/`u`/`v`를 마지막 축
기준으로 펼쳐 Kaiming-uniform(`a=sqrt(5)`)로 초기화하고, `w_gate`는
`±1/sqrt(d_h)` 균등 분포로 초기화한다. 게이트의 시그모이드 입력이 과도하게
포화되지 않도록 한 선택이다.

---

## 7. 참고문헌

- **Flash Multi-Head FFN (arXiv 2025)**:
  Zhang, Hu, Li, Wu, Tu, "Flash Multi-Head FFN"
  — <https://arxiv.org/abs/2512.06989>.
  본 `MultiHeadFFN`은 이 논문의 아키텍처(Eq. 10–14)에 대한 하드웨어 독립 순수
  PyTorch 재구현이다.
- **Hourglass / shape convention**:
  Liao, Chen, Yi, Shiu, "Revisiting the Shape Convention of Transformer
  Language Models" — <https://arxiv.org/abs/2602.06471>;
  Chen, Lee, Liao, Shiu, "Rethinking the Shape Convention of an MLP"
  (OpenReview `bUtLHJn90a`) — <https://arxiv.org/abs/2510.01796>.
- **SwiGLU**: Shazeer, "GLU Variants Improve Transformer" (2020)
  — <https://arxiv.org/abs/2002.05202>. 두 블록의 sub-MLP/sub-network가
  공유하는 게이트형 활성.
- **FlashAttention**: Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness" (2022)
  — <https://arxiv.org/abs/2205.14135>. MultiHeadFFN 원논문의 fused 커널이
  따르는 I/O-aware 설계의 원형.
