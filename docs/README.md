# ypsilon-torch — Documentation / 문서

## 한국어 문서

- [**Bilinear MLP 블록 (AFBO 스타일)**](bilinear_mlp.ko.md) — AFBO의
  spatial-channel factorization 위에 Bilinear MLP를 얹은
  `AsymmetricSpatialChannelFactorizedBilinearMLP` 블록의 수학적 정의,
  API 레퍼런스, 사용 예제, 하이퍼파라미터 가이드, 구현 노트.
- [**Robust Statistics (함수형 유틸리티)**](robust_stats.ko.md) — Fréchet
  median/medoid, arsinh 공간 통계량의 수학적 정의, API 레퍼런스, 사용 예제.
  L_p 특화 및 임의 거리 함수 지원.
- [**Normalization 블록**](normalizations.ko.md) — RobustLayerNorm,
  AsinhMeanLayerNorm 및 complex 확장의 수학적 정의, API 레퍼런스, 사용
  예제, 하이퍼파라미터 가이드.
- [**N-dimensional Pooling 블록**](nd_pooling.ko.md) — 1D/2D/3D 통합 지원
  robust pooling (Asinh, Fréchet median/medoid, Root-\*-Square), real/complex
  도메인, L_p 및 general 거리 함수 지원.
- [**활성화 함수 (확장)**](activations.ko.md) — ArSinh, ThAShGLU(real) 및
  ComplexThASh, ComplexHGLU, ZReLU, CReLU(complex)의 수학적 정의, API
  레퍼런스, 사용 예제. 기존 HGLU/ThASh의 complex 대응물 포함.
- [**Robust 위치·스케일 추정 (확장)**](robust_location.ko.md) — Huber/Tukey
  biweight location, trimmed/Winsorized mean, MAD, biweight midvariance 및
  이를 활용한 HuberLayerNorm, TrimmedLayerNorm.
- [**Regularization 블록**](regularizations.ko.md) — GaussianNoise,
  ArsinhGaussianNoise(arsinh 공간 scale-aware 노이즈)의 수학적 정의, API
  레퍼런스, 사용 예제, 하이퍼파라미터 가이드.
- [**Feed-Forward(FFN) 대체 블록**](feedforward.ko.md) — HourglassFFN
  (wide-narrow-wide 스택), MultiHeadFFN(Flash Multi-Head FFN의 하드웨어
  독립 순수 PyTorch 재구현)의 수학적 정의, 아키텍처, API, 구현 노트.
- [**Mixture (희소·조건부 연산) 블록**](mixture.ko.md) — Mixture-of-Depths,
  Mixture-of-Depths-and-Experts, Mixture-of-Hidden-Dimensions,
  Mixture-of-Lookup-Experts, Attractor Patch Networks 및 torchutils.auxloss
  기반 reg_loss 통합.

## English Documentation

- [**Bilinear MLP Block (AFBO-style)**](bilinear_mlp.en.md) — mathematical
  formulation, API reference, usage examples, hyperparameter guide, and
  implementation notes for the
  `AsymmetricSpatialChannelFactorizedBilinearMLP` block, which combines
  AFBO's spatial-channel factorization with the Bilinear MLP formulation.
- [**Robust Statistics (Functional Utilities)**](robust_stats.en.md) —
  mathematical definitions, API reference, and usage examples for Fréchet
  median/medoid and arsinh-space statistics. L_p specialisations and
  arbitrary distance function support.
- [**Normalization Blocks**](normalizations.en.md) — mathematical
  definitions, API reference, usage examples, and hyperparameter guide for
  RobustLayerNorm, AsinhMeanLayerNorm, and their complex extensions.
- [**N-dimensional Pooling Blocks**](nd_pooling.en.md) — unified 1D/2D/3D
  robust pooling (Asinh, Fréchet median/medoid, Root-\*-Square), real/complex
  domains, L_p and general distance function support.
- [**Activation Functions (extensions)**](activations.en.md) — mathematical
  definitions, API reference, and usage for ArSinh, ThAShGLU (real) and
  ComplexThASh, ComplexHGLU, ZReLU, CReLU (complex), including complex
  counterparts of the existing HGLU/ThASh.
- [**Robust Location & Scale Estimators (extensions)**](robust_location.en.md) —
  Huber / Tukey-biweight location, trimmed / Winsorized mean, MAD, biweight
  midvariance, and the HuberLayerNorm / TrimmedLayerNorm blocks built on them.
- [**Regularization Blocks**](regularizations.en.md) — mathematical
  definitions, API reference, usage, and hyperparameter guide for
  GaussianNoise and ArsinhGaussianNoise (arsinh-space, scale-aware noise).
- [**Feed-Forward (FFN) Replacement Blocks**](feedforward.en.md) — HourglassFFN
  (wide-narrow-wide stack) and MultiHeadFFN (a hardware-independent pure-PyTorch
  re-implementation of Flash Multi-Head FFN): formulation, architecture, API,
  implementation notes.
- [**Mixture (Sparse / Conditional-Computation) Blocks**](mixture.en.md) —
  Mixture-of-Depths, Mixture-of-Depths-and-Experts, Mixture-of-Hidden-Dimensions,
  Mixture-of-Lookup-Experts, Attractor Patch Networks, and torchutils.auxloss-
  backed reg_loss integration.

---

Every `*.ko.md` has a 1:1 English sibling `*.en.md` with identical section
ordering and identical equations / code blocks, so cross-language linking is
trivial (replace the `.ko.` / `.en.` infix).

모든 `*.ko.md` 문서에는 동일한 섹션 순서와 동일한 수식·코드 블록을 가진
`*.en.md` 쌍둥이 문서가 있다 (확장자 infix를 `.ko.` ↔ `.en.`로 교체).
