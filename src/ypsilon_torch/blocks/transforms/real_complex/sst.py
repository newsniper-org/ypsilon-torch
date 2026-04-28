import torch
from typing import TypeVar, Dict, Tuple, Generic, TypeVarTuple, get_args, TypedDict, Unpack

try:
    from typing import override
except:
    from typing_extensions import override

from functools import reduce

from collections.abc import Sequence
from ypsilon_torch import NonLearnableProcessorBase, NonLearnableSynchronizedProcessorBase, FPDTypeIdx, get_float_dtype, get_complex_dtype
from ypsilon_torch.transforms import InvertibleTransformsBase

class SynchronizedGenericSSTCache(TypedDict):
    # 역변환을 위한 상태 캐시: { cache_key: (reassigned_bins, original_z) }
    sync_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]]

    # 캐시를 조회하기 위한 키
    cache_key: str

class SynchronizedGenericSST[T: InvertibleTransformsBase['I'], I: InvertibleTransformsBase[T]](NonLearnableSynchronizedProcessorBase[SynchronizedGenericSSTCache]):
    """
    N개의 다중 입력을 동기화하여 단일 Squeezing 상태를 공유하는 특수 SST 컴포넌트
    Python 3.13의 제네릭 타이핑을 적극 활용하여 설계되었습니다.
    """
    @override
    def __init__(self, base_transform: T, dtype_idx: FPDTypeIdx, freq_dim_groups: int = 1):
        super().__init__()
        self.base_transform: T = base_transform
        self.freq_dim_groups: int = freq_dim_groups
        self.dtype_idx: FPDTypeIdx = dtype_idx

    @override
    def is_valid_input(self, x: torch.Tensor) -> bool:
        return self.base_transform.is_valid_input(x)

    @override
    def transform(self, *xs: torch.Tensor, **caches: Unpack[SynchronizedGenericSSTCache]) -> tuple[torch.Tensor, ...]:
        N = len(xs)
        if N == 0:
            return ()

        # 1. 모든 입력에 대해 베이스 변환 수행
        zs = tuple(self.base_transform.transform(x) for x in xs)
        
        B, C_total = zs[0].shape[:2]
        spatial_shape = tuple(zs[0].shape[2:])
        freq_bins = C_total // self.freq_dim_groups
        
        z_reshaped_s = tuple(z.view(B, self.freq_dim_groups, freq_bins, *spatial_shape) for z in zs)

        # 2. 다중 모달리티 융합을 위한 통합 위상 도함수(IF) 계산
        shared_if_sum = torch.zeros_like(z_reshaped_s[0], dtype=get_float_dtype(self.dtype_idx))
        total_mag = torch.zeros_like(shared_if_sum, dtype=get_float_dtype(self.dtype_idx))

        for z_res in z_reshaped_s:
            phase = torch.angle(z_res)
            mag = torch.abs(z_res)
            if_sum = torch.zeros_like(phase, dtype=get_float_dtype(self.dtype_idx))
            
            for dim in range(3, z_res.ndim):
                if_sum += torch.abs(torch.gradient(phase, dim=dim)[0])
            
            # 진폭이 큰(에너지가 강한) 신호의 순간 주파수에 더 높은 가중치를 부여
            shared_if_sum += if_sum * mag
            total_mag += mag
            
        eps = torch.finfo(get_float_dtype(self.dtype_idx)).eps
        shared_if_sum = torch.where(
            total_mag.abs() > eps,
            shared_if_sum / total_mag,
            torch.zeros_like(shared_if_sum),
        )

        # 3. 통합된 재배치 인덱스 생성
        if_min = shared_if_sum.min(dim=2, keepdim=True)[0]
        if_max = shared_if_sum.max(dim=2, keepdim=True)[0]
        if_den = if_max - if_min
        if_norm = torch.where(
            if_den.abs() > eps,
            (shared_if_sum - if_min) / if_den,
            torch.zeros_like(shared_if_sum - if_min)
        )

        shared_reassigned_bins = torch.round(if_norm * (freq_bins - 1)).long()
        shared_reassigned_bins = torch.clamp(shared_reassigned_bins, 0, freq_bins - 1)

        # 4. 동일한 기준(shared_reassigned_bins)으로 모든 Z 일괄 Squeezing
        original_z_sum = reduce(lambda a, b: a + b, z_reshaped_s,
            torch.zeros_like(z_reshaped_s[0], dtype=get_float_dtype(self.dtype_idx)) # 역변환 가중치용 통합 Z
        )
        
        def _squeeze(zrs: tuple[torch.Tensor, ...]):
            for z_res in zrs:
                squeezed_z = torch.zeros_like(z_res, dtype=get_complex_dtype(self.dtype_idx))
                squeezed_z.scatter_add_(dim=2, index=shared_reassigned_bins, src=z_res)
                yield squeezed_z.view(B, C_total, *spatial_shape)

        squeezed_zs = tuple(squeezed_z for squeezed_z in _squeeze(z_reshaped_s))
        

        # 5. 역변환기(Inverse)를 위한 단일 동기화 상태 캐싱
        sync_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = caches['sync_cache']
        cache_key: str = caches['cache_key']
        if len(sync_cache) > 8:
            sync_cache.pop(next(iter(sync_cache)))
        # 개별 Z가 아닌, 평균적인 Z의 분포(original_z_sum)를 저장하여 단일 출력 W 역산에 사용
        sync_cache[cache_key] = (shared_reassigned_bins, original_z_sum)

        return squeezed_zs
    
    def get_inverse_transform(self) -> 'InverseSynchronizedGenericSST[T, I]':
        return InverseSynchronizedGenericSST[T, I](self.base_transform.get_inverse_transform(), self.dtype_idx, self.freq_dim_groups)


class InverseSynchronizedGenericSST[T: InvertibleTransformsBase['I'], I: InvertibleTransformsBase[T]](NonLearnableSynchronizedProcessorBase[SynchronizedGenericSSTCache]):
    """
    동기화된 단일 캐시를 활용하여 신경망의 융합 출력 W를 역변환합니다.
    """
    @override
    def __init__(self, base_inverse_transform: I, dtype_idx: FPDTypeIdx, freq_dim_groups: int = 1):
        super().__init__()
        self.base_inverse_transform: I = base_inverse_transform
        self.freq_dim_groups: int = freq_dim_groups
        self.dtype_idx: FPDTypeIdx = dtype_idx

    @override
    def is_valid_input(self, z: torch.Tensor) -> bool:
        return self.base_inverse_transform.is_valid_input(z)

    def _transform_each(self, w: torch.Tensor, shared_reassigned_bins: torch.Tensor, original_z_sum: torch.Tensor) -> torch.Tensor:
        B, C_total = w.shape[:2]
        spatial_shape = tuple(w.shape[2:])
        freq_bins = C_total // self.freq_dim_groups
        
        w_reshaped = w.view(B, self.freq_dim_groups, freq_bins, *spatial_shape)

        # 1. 융합 전 통합 Z의 Squeezed 합산 복원
        original_squeezed = torch.zeros_like(original_z_sum, dtype=get_complex_dtype(self.dtype_idx))
        original_squeezed.scatter_add_(dim=2, index=shared_reassigned_bins, src=original_z_sum)
        
        # 2. 비율 가중치(Weight) 계산 
        den = torch.gather(original_squeezed, dim=2, index=shared_reassigned_bins)
        eps = torch.finfo(get_float_dtype(self.dtype_idx)).eps
        weight = torch.where(
            den.abs() > eps,
            original_z_sum / den,
            torch.zeros_like(original_z_sum),
        )

        # 3. 융합 결과물 W 역분배 (Redistribution)
        gathered_w = torch.gather(w_reshaped, dim=2, index=shared_reassigned_bins)
        unsqueezed_w = gathered_w * weight

        unsqueezed_w = unsqueezed_w.view(B, C_total, *spatial_shape)
        
        # 4. 베이스 역변환
        final_y = self.base_inverse_transform.transform(unsqueezed_w)
        
        return final_y

    @override
    def transform(self, *ws: torch.Tensor, **caches: Unpack[SynchronizedGenericSSTCache]) -> tuple[torch.Tensor, ...]:
        sync_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = caches['sync_cache']
        cache_key: str = caches['cache_key']
        if cache_key not in sync_cache.keys():
            raise RuntimeError(f"동기화 캐시({cache_key}) 누락. 전방 transform_multiple()이 선행되어야 합니다.")

        shared_reassigned_bins, original_z_sum = sync_cache.pop(cache_key)
        
        return tuple(map(lambda w: self._transform_each(w, shared_reassigned_bins, original_z_sum),ws))
    
    def get_inverse_transform(self) -> SynchronizedGenericSST[T, I]:
        return SynchronizedGenericSST[T, I](self.base_inverse_transform.get_inverse_transform(), self.dtype_idx, self.freq_dim_groups)




        