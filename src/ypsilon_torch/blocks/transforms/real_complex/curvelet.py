import math
import torch
from typing import Dict, List, Tuple

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ypsilon_torch import NonLearnableProcessorBase, FPDTypeIdx, get_complex_dtype

from ypsilon_torch.transforms import InvertibleTransformsBase

class FastCurveletTransform2D(InvertibleTransformsBase['InverseFastCurveletTransform2D']):
    """
    2D Fast Discrete Curvelet Transform
    주파수 도메인의 코로나 및 쐐기(Wedge) 마스킹을 통한 고선명도 곡선 에지 추출
    """
    @override
    def __init__(self, scales: int = 3, angles_coarsest: int = 8, dtype_idx: FPDTypeIdx = 64):
        super().__init__()
        self.scales: int = scales
        self.angles_coarsest: int = angles_coarsest
        self.D: int = 2
        self._mask_cache: Dict[Tuple[int, int, torch.device], List[torch.Tensor]] = {}
        self.dtype_idx: FPDTypeIdx = dtype_idx

    @override
    def is_valid_input(self, x: torch.Tensor) -> bool:
        return (not x.dtype.is_complex) and (x.ndim == 4)

    def _get_wrapping_masks(self, H: int, W: int, device: torch.device) -> List[torch.Tensor]:
        cache_key = (H, W, device)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        masks: List[torch.Tensor] = []
        cy, cx = H // 2, W // 2
        
        # 2D 주파수 그리드 생성
        Y, X = torch.meshgrid(
            torch.arange(H, device=device) - cy, 
            torch.arange(W, device=device) - cx, 
            indexing='ij'
        )
        R = torch.sqrt(Y**2 + X**2)
        Theta = torch.atan2(Y, X)

        max_R = math.sqrt(cy*cy + cx*cx)
        
        # 1. 최저 주파수 대역 (중앙 코로나, Low-pass)
        r_mask = R < (max_R / (2**(self.scales - 1)))
        masks.append(r_mask.to(get_complex_dtype(self.dtype_idx)))

        # 2. 세부 주파수 대역 및 각도별 쐐기(Wedge) 분할
        for s in range(1, self.scales):
            r_inner = max_R / (2**(self.scales - s))
            r_outer = max_R / (2**(self.scales - s - 1))
            num_angles = self.angles_coarsest * (2**(s - 1))
            angle_step = 2 * math.pi / num_angles

            for a in range(num_angles):
                angle_center = -math.pi + a * angle_step
                # 위상 랩어라운드(Wrap-around)를 고려한 각도 차이 계산
                angle_diff = torch.abs(torch.atan2(
                    torch.sin(Theta - angle_center), 
                    torch.cos(Theta - angle_center)
                ))
                
                if s == self.scales - 1:
                    w_mask = (R >= r_inner) & (R <= r_outer) & (angle_diff <= angle_step / 2)
                else:
                    w_mask = (R >= r_inner) & (R < r_outer) & (angle_diff <= angle_step / 2)
                masks.append(w_mask.to(get_complex_dtype(self.dtype_idx)))

        if len(self._mask_cache) > 16:
            self._mask_cache.pop(next(iter(self._mask_cache)))
        self._mask_cache[cache_key] = masks
        
        return masks

    @override
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_c = x.to(get_complex_dtype(self.dtype_idx))

        # FFT 후 주파수 중심을 텐서 중앙으로 이동
        X_f = torch.fft.fftshift(torch.fft.fft2(x_c, dim=(-2, -1)), dim=(-2, -1))

        masks = self._get_wrapping_masks(H, W, x.device)
        
        Z_list: List[torch.Tensor] = []

        # 각 타일 마스크를 적용하여 역변환
        for mask in masks:
            band_f = X_f * mask.unsqueeze(0).unsqueeze(0)
            band_f_ishift = torch.fft.ifftshift(band_f, dim=(-2, -1))
            band_spatial = torch.fft.ifft2(band_f_ishift, dim=(-2, -1))
            Z_list.append(band_spatial)

        # 추출된 모든 대역의 곡선 에지 피처를 채널 차원으로 병합
        result = torch.cat(Z_list, dim=1)
        return result
    
    @override
    def get_inverse_transform(self) -> 'InverseFastCurveletTransform2D':
        return InverseFastCurveletTransform2D(self)


class InverseFastCurveletTransform2D(InvertibleTransformsBase[FastCurveletTransform2D]):
    """
    2D Fast Discrete Curvelet Transform 역변환
    """
    @override
    def __init__(self, forward_transform_instance: FastCurveletTransform2D):
        super().__init__()
        self.forward_t: FastCurveletTransform2D = forward_transform_instance
        self.dtype_idx = self.forward_t.dtype_idx

    @override
    def is_valid_input(self, z: torch.Tensor) -> bool:
        return z.dtype.is_complex and (z.ndim == 4)

    @override
    def transform(self, z: torch.Tensor) -> torch.Tensor:
        B, C_total, H, W = z.shape
        masks = self.forward_t._get_wrapping_masks(H, W, z.device)
        num_masks = len(masks)
        
        if C_total % num_masks != 0:
            raise ValueError(f"입력 채널({C_total})이 타일 마스크 수({num_masks})와 일치하지 않습니다.")
            
        C = C_total // num_masks
        
        # 병합되었던 채널을 타일 마스크 갯수만큼 분리
        z_reshaped = z.view(B, num_masks, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        
        # 역산을 위한 누산기 텐서
        X_f_recon = torch.zeros((B, C, H, W), dtype=get_complex_dtype(self.dtype_idx), device=z.device)
        
        for i, mask in enumerate(masks):
            band_spatial = z_reshaped[:, :, i, :, :]
            band_f_ishift = torch.fft.fft2(band_spatial, dim=(-2, -1))
            band_f = torch.fft.fftshift(band_f_ishift, dim=(-2, -1))
            
            # 해당 주파수 대역의 에너지를 원래 위치로 누산
            X_f_recon += band_f * mask.unsqueeze(0).unsqueeze(0)
            
        x_recon = torch.fft.ifft2(torch.fft.ifftshift(X_f_recon, dim=(-2, -1)), dim=(-2, -1))
        
        return x_recon.real
    
    @override
    def get_inverse_transform(self) -> FastCurveletTransform2D:
        return self.forward_t