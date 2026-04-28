import torch
import torch.nn as nn
from ypsilon_torch import NonLearnableProcessorBase, FPDTypeIdx, get_complex_dtype, get_float_dtype
from ypsilon_torch.transforms import InvertibleTransformsBase

try:
    from typing import override
except ImportError:
    from typing_extension import override

class RieszTransform(InvertibleTransformsBase['InverseRieszTransform']):
    """
    N차원 Riesz Transform (다차원 일반화 힐베르트 변환)
    입력: (B, C, D1, ..., Dn) Real Tensor
    출력: (B, C * N, D1, ..., Dn) Complex Tensor
    """
    def __init__(self, D: int, dtype_idx: FPDTypeIdx):
        super().__init__()
        self.D: int = D
        self.dtype_idx: FPDTypeIdx = dtype_idx
        # (spatial_shape, device)를 키로 사용하는 승수 캐시
        self._cache: dict[tuple[tuple[int, ...], torch.device], torch.Tensor] = {}

    def is_valid_input(self, x: torch.Tensor) -> bool:
        return (not x.dtype.is_complex) and (x.ndim >= self.D + 2)

    def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=get_float_dtype(self.dtype_idx)).to(dtype=get_complex_dtype(self.dtype_idx))

    def _get_multipliers(self, spatial_shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        cache_key = (spatial_shape, device)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # N차원 주파수 그리드 생성
        grids = [torch.fft.fftfreq(s, device=device, dtype=get_float_dtype(self.dtype_idx)) for s in spatial_shape]
        mesh = torch.meshgrid(*grids, indexing='ij')
        
        # 유클리드 노름 계산 및 0으로 나누기 방지
        norm = torch.sqrt(sum(m**2 for m in mesh))
        norm[norm == 0] = 1.0 

        # Riesz 승수: -i * (w_k / |w|)
        multipliers = torch.stack([-1j * (m / norm) for m in mesh], dim=0)
        
        # DC 성분(원점)은 0으로 처리
        zero_idx = (0,) * self.D
        for i in range(self.D):
            multipliers[i][zero_idx] = 0.0

        if len(self._cache) > 32:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = multipliers
        return multipliers

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = tuple(x.shape[2: 2 + self.D])
        dims = tuple(range(2, 2 + self.D))
        B, C = x.shape[:2]

        x_c = self._to_complex(x)
        X_f = torch.fft.fftn(x_c, dim=dims)

        multipliers = self._get_multipliers(spatial_shape, x.device)
        Z_list = []

        # 각 차원별 Riesz 성분을 계산하여 복소수 텐서 생성 (X + i * R_k)
        for d in range(self.D):
            M_d = multipliers[d].unsqueeze(0).unsqueeze(0)
            R_d_f = X_f * M_d
            R_d = torch.fft.ifftn(R_d_f, dim=dims).real.to(dtype=x.dtype)
            Z_list.append(torch.complex(x, R_d))

        # 채널 차원을 따라 결합: (B, C * D, D1, ..., Dn)
        return torch.cat(Z_list, dim=1)
    
    @override
    def get_inverse_transform(self) -> 'InverseRieszTransform':
        return InverseRieszTransform(self.D, self.dtype_idx)


class InverseRieszTransform(InvertibleTransformsBase[RieszTransform]):
    """
    N차원 Riesz Transform의 역변환
    입력: (B, C * N, D1, ..., Dn) Complex Tensor
    출력: (B, C, D1, ..., Dn) Real Tensor
    """
    def __init__(self, D: int, dtype_idx: FPDTypeIdx):
        super().__init__()
        self.D = D
        self.dtype_idx: FPDTypeIdx = dtype_idx

    def is_valid_input(self, z: torch.Tensor) -> bool:
        return z.dtype.is_complex and (z.ndim >= self.D + 2)

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        B, C_total = z.shape[:2]
        spatial_shape = tuple(z.shape[2:])
        
        if C_total % self.D != 0:
            raise ValueError(f"입력 채널 수({C_total})는 공간 차원 수({self.D})의 배수여야 합니다.")
            
        C = C_total // self.D
        
        # 확장되었던 채널을 분리: (B, C, D, D1, ..., Dn)
        z_reshaped = z.view(B, self.D, C, *spatial_shape).permute(0, 2, 1, *range(3, 3 + self.D)).contiguous()
        
        # 각 차원별로 계산된 복소수 텐서들의 실수부는 모두 동일한 원본 신호 X입니다.
        # 수치적 안정성을 위해 D차원에 대한 평균을 취합니다.
        recon_x = z_reshaped.real.mean(dim=2)
        
        return recon_x
    
    @override
    def get_inverse_transform(self) -> RieszTransform:
        return RieszTransform(self.D, self.dtype_idx)