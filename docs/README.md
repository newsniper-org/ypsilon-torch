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

---

Every `*.ko.md` has a 1:1 English sibling `*.en.md` with identical section
ordering and identical equations / code blocks, so cross-language linking is
trivial (replace the `.ko.` / `.en.` infix).

모든 `*.ko.md` 문서에는 동일한 섹션 순서와 동일한 수식·코드 블록을 가진
`*.en.md` 쌍둥이 문서가 있다 (확장자 infix를 `.ko.` ↔ `.en.`로 교체).
