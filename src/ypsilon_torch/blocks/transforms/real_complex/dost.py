from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Literal, final, Self

import torch
import torch.nn as nn

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ypsilon_torch import NonLearnableProcessorBase, FPDTypeIdx, COMPLEX_DTYPES_DICT, FLOAT_DTYPES_DICT
from ypsilon_torch.blocks.transforms import InvertibleTransformsBase

from collections.abc import Sequence


def get_dyadic_partitions(N: int) -> list[tuple[int, int]]:
    """Dyadic partitions over the FFT index axis [0, N)."""
    if N <= 0:
        raise ValueError("N must be positive")
    N_eff = N if (N % 2 == 0) else (N - 1)
    parts: list[tuple[int, int]] = [(0, 1)]

    k = 1
    half = N_eff // 2
    while k < half:
        end = min(2 * k, half)
        parts.append((k, end))
        k *= 2

    if half < N_eff:
        parts.append((half, N_eff))
    if N_eff != N:
        parts.append((N_eff, N))
    return parts


@lru_cache(maxsize=256)
def _band_slices_cached(spatial_shape: tuple[int, ...]) -> tuple[tuple[slice, ...], ...]:
    per_dim = [get_dyadic_partitions(s) for s in spatial_shape]
    bands = list(itertools.product(*per_dim))
    out: list[tuple[slice, ...]] = []
    for band in bands:
        out.append(tuple(slice(int(st), int(en)) for (st, en) in band))
    return tuple(out)

class DynamicDOSTBase[I: DynamicDOSTBase[Self]](InvertibleTransformsBase[I], ABC):
    """Common base for dynamic DOST/IDOST with cached tensors and metadata."""

    @abstractmethod
    @override
    def __init__(self, D: int):
        super().__init__()
        self.D: int = D
        
        # 해상도와 디바이스를 키로 사용하는 내부 캐시 딕셔너리
        # 구조: { (spatial_shape, device): {"mask": Tensor, "band_start": Tensor, "band_end": Tensor} }
        self._tensor_cache: dict[tuple[tuple[int, ...], torch.device], dict[str, torch.Tensor]] = {}

    def _get_cached_buffers(
        self, spatial_shape: tuple[int, ...], device: torch.device
    ) -> tuple[int, tuple[tuple[slice, ...], ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        주어진 해상도와 디바이스에 대한 메타데이터와 텐서들을 반환합니다.
        캐시 히트 시 기존 최적화와 동일하게 메모리 할당 없이 즉시 반환합니다.
        """
        band_slices = _band_slices_cached(spatial_shape)
        num_bands = len(band_slices)
        
        cache_key = (spatial_shape, device)
        if cache_key in self._tensor_cache:
            buffers = self._tensor_cache[cache_key]
            return num_bands, band_slices, buffers["mask"], buffers["band_start"], buffers["band_end"]

        # 캐시에 없는 경우 새롭게 할당 및 생성 (최초 1회만 실행됨)
        bs = torch.empty((num_bands, self.D), dtype=torch.int64, device=device)
        be = torch.empty((num_bands, self.D), dtype=torch.int64, device=device)
        for b, sl in enumerate(band_slices):
            for d, s in enumerate(sl):
                bs[b, d] = int(s.start or 0)
                be[b, d] = int(s.stop)

        # Slice assignment 연산을 통한 빠른 dense mask 생성
        mask = torch.zeros((num_bands, *spatial_shape), dtype=torch.bool, device=device)
        for b, sl in enumerate(band_slices):
            mask[(b,) + sl] = True

        # OOM(Out of Memory) 방지를 위해 캐시 사이즈를 제한 (예: 32개 해상도)
        if len(self._tensor_cache) > 32:
            self._tensor_cache.pop(next(iter(self._tensor_cache)))

        self._tensor_cache[cache_key] = {
            "mask": mask,
            "band_start": bs,
            "band_end": be
        }

        return num_bands, band_slices, mask, bs, be

    @final
    def _fft_dims(self) -> tuple[int, ...]:
        return tuple(range(2, 2 + self.D))


class DOST(DynamicDOSTBase['IDOST']):
    """Discrete Orthogonal Stockwell Transform (Dynamic Resolution)."""

    @override
    def __init__(self, D: int):
        super().__init__(D)

    @override
    def is_valid_input(self, x: torch.Tensor) -> bool:
        return (not x.dtype.is_complex) and (x.ndim >= self.D + 2)

    def _to_complex(self, x: torch.Tensor, dtype_idx: FPDTypeIdx | None = None) -> torch.Tensor:
        if dtype_idx is None:
            if x.dtype == torch.float64:
                return x.to(dtype=torch.complex128)
            if x.dtype == torch.float16:
                return x.to(dtype=torch.complex32)
            return x.to(dtype=torch.float32).to(dtype=torch.complex64)
        else:
            return x.to(dtype=FLOAT_DTYPES_DICT[dtype_idx]).to(dtype=COMPLEX_DTYPES_DICT[dtype_idx])

    @override
    def transform(self, x: torch.Tensor, dtype_idx: FPDTypeIdx | None = None) -> torch.Tensor:
        dims = self._fft_dims()
        spatial_shape = tuple(x.shape[2 : 2 + self.D])
        
        x_c = self._to_complex(x, dtype_idx)
        f_x = torch.fft.fftn(x_c, dim=dims).unsqueeze(2)

        # 동적 캐시에서 정보 획득
        num_bands, _, mask, _, _ = self._get_cached_buffers(spatial_shape, f_x.device)

        band_freq = f_x * mask.unsqueeze(0).unsqueeze(0)
        band_time = torch.fft.ifftn(band_freq, dim=tuple(dim+1 for dim in dims))

        B, C = x.shape[:2]
        return band_time.reshape(B, C * num_bands, *spatial_shape)
    
    @override
    def get_inverse_transform(self) -> 'IDOST':
        return IDOST(self.D)


class IDOST(DynamicDOSTBase[DOST]):
    """Inverse DOST (Dynamic Resolution)."""

    @override
    def __init__(self, D: int):
        super().__init__(D)

    @override
    def is_valid_input(self, z: torch.Tensor) -> bool:
        return z.dtype.is_complex and (z.ndim >= self.D + 2)

    def transform(
        self,
        z: torch.Tensor,
        *,
        strategy: Literal["auto", "dense", "sparse"] = "auto",
        band_block: int = 64,
        dtype_idx: FPDTypeIdx | None = None
    ) -> torch.Tensor:
        dims = self._fft_dims()
        spatial_shape = tuple(z.shape[2 : 2 + self.D])
        
        # 동적 캐시에서 정보 획득
        num_bands, band_slices, mask, _, _ = self._get_cached_buffers(spatial_shape, z.device)

        B, C_expanded = z.shape[:2]
        if C_expanded % num_bands != 0:
            raise RuntimeError("Invalid DOST band structure")
        C = C_expanded // num_bands
        z = z.view(B, C, num_bands, *spatial_shape)

        if strategy == "auto":
            strategy = "sparse"

        if strategy == "dense":
            z_f = torch.fft.fftn(z, dim=tuple(dim+1 for dim in dims))
            f_recon = torch.sum(z_f * mask.unsqueeze(0).unsqueeze(0), dim=2)
            recon = torch.fft.ifftn(f_recon, dim=dims)
            
            if dtype_idx is None:
                return recon.real
            else:
                return recon.real.to(FLOAT_DTYPES_DICT[dtype_idx])

        # sparse: slice-accumulate
        f_recon = torch.zeros((B, C, *spatial_shape), device=z.device, dtype=z.dtype)
        blk = int(band_block)
        if blk <= 0:
            blk = num_bands
        blk = min(blk, num_bands)

        for b0 in range(0, num_bands, blk):
            b1 = min(b0 + blk, num_bands)
            z_blk = z[:, :, b0:b1].contiguous()
            zf_blk = torch.fft.fftn(z_blk, dim=tuple(dim+1 for dim in dims))
            for i, band in enumerate(range(b0, b1)):
                sl = band_slices[band]
                f_recon[(...,) + sl] += zf_blk[:, :, i][(...,) + sl]

        recon = torch.fft.ifftn(f_recon, dim=dims)

        if dtype_idx is None:
            return recon.real
        else:
            return recon.real.to(FLOAT_DTYPES_DICT[dtype_idx])
    
    @override
    def get_inverse_transform(self) -> DOST:
        return DOST(self.D)