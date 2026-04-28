import math
import torch
from typing import Dict, List, Tuple

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ypsilon_torch import NonLearnableProcessorBase, FPDTypeIdx, get_complex_dtype, get_float_dtype
from ypsilon_torch.transforms import InvertibleTransformsBase


class FastSurfaceletTransform3D(InvertibleTransformsBase['InverseFastSurfaceletTransform3D']):
    """
    3D Fast Discrete Surfacelet Transform
    3차원 구면 좌표계를 활용한 표면/볼륨 에지 최적화 타일링
    """

    @override
    def __init__(self, scales: int = 3, angles_coarsest: int = 8, dtype_idx: FPDTypeIdx = 64):
        super().__init__()
        self.scales: int = scales
        self.angles_coarsest: int = angles_coarsest
        self.D: int = 3
        self._mask_cache: Dict[Tuple[int, int, int, torch.device], List[torch.Tensor]] = {}
        self.dtype_idx: FPDTypeIdx = dtype_idx

    @override
    def is_valid_input(self, x: torch.Tensor) -> bool:
        return (not x.dtype.is_complex) and (x.ndim == 5)

    def _get_wrapping_masks(self, D_dim: int, H: int, W: int, device: torch.device) -> List[torch.Tensor]:
        cache_key = (D_dim, H, W, device)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        masks: List[torch.Tensor] = []
        cd, cy, cx = D_dim // 2, H // 2, W // 2

        Z, Y, X = torch.meshgrid(
            torch.arange(D_dim, device=device) - cd,
            torch.arange(H, device=device) - cy,
            torch.arange(W, device=device) - cx,
            indexing='ij'
        )

        # 3차원 구면 좌표
        R = torch.sqrt(Z**2 + Y**2 + X**2)
        Theta = torch.remainder(torch.atan2(Y, X) + 2.0 * math.pi, 2.0 * math.pi)  # [0, 2pi)
        Phi = torch.acos(torch.clamp(Z / (R + 1e-8), -1.0, 1.0))                    # [0, pi]

        max_R = math.sqrt(cd * cd + cy * cy + cx * cx)
        complex_dtype = get_complex_dtype(self.dtype_idx)

        # 1) 최저 주파수 대역 (중앙 구체)
        lowpass_radius = max_R / (2 ** (self.scales - 1))
        lowpass_mask = (R < lowpass_radius)
        masks.append(lowpass_mask.to(dtype=complex_dtype))

        # 2) 나머지 대역: 반경 shell + 구면각 bin을 floor 인덱싱으로 정확 분할
        for s in range(1, self.scales):
            r_inner = max_R / (2 ** (self.scales - s))
            r_outer = max_R / (2 ** (self.scales - s - 1))

            num_theta = self.angles_coarsest * (2 ** (s - 1))
            num_phi = max(1, num_theta // 2)

            if s == self.scales - 1:
                shell_mask = (R >= r_inner) & (R <= r_outer)
            else:
                shell_mask = (R >= r_inner) & (R < r_outer)

            theta_step = 2.0 * math.pi / num_theta
            phi_step = math.pi / num_phi

            # phi=pi가 마지막 bin에 안정적으로 들어가도록 clamp
            phi_clamped = torch.clamp(Phi, min=0.0, max=math.pi - 1e-8)

            theta_idx = torch.floor(Theta / theta_step).to(torch.long).clamp_(0, num_theta - 1)
            phi_idx = torch.floor(phi_clamped / phi_step).to(torch.long).clamp_(0, num_phi - 1)

            for t in range(num_theta):
                for p in range(num_phi):
                    w_mask = shell_mask & (theta_idx == t) & (phi_idx == p)
                    masks.append(w_mask.to(dtype=complex_dtype))

        if len(self._mask_cache) > 8:
            self._mask_cache.pop(next(iter(self._mask_cache)))
        self._mask_cache[cache_key] = masks
        return masks

    @override
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D_dim, H, W = x.shape
        x_c = x.to(dtype=get_complex_dtype(self.dtype_idx))

        X_f = torch.fft.fftshift(torch.fft.fftn(x_c, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        masks = self._get_wrapping_masks(D_dim, H, W, x.device)

        Z_list: List[torch.Tensor] = []

        for mask in masks:
            band_f = X_f * mask.unsqueeze(0).unsqueeze(0)
            band_f_ishift = torch.fft.ifftshift(band_f, dim=(-3, -2, -1))
            band_spatial = torch.fft.ifftn(band_f_ishift, dim=(-3, -2, -1))
            Z_list.append(band_spatial)

        return torch.cat(Z_list, dim=1)

    @override
    def get_inverse_transform(self) -> 'InverseFastSurfaceletTransform3D':
        return InverseFastSurfaceletTransform3D(self)


class InverseFastSurfaceletTransform3D(InvertibleTransformsBase[FastSurfaceletTransform3D]):
    """
    3D Fast Discrete Surfacelet Transform 역변환
    """

    @override
    def __init__(self, forward_transform_instance: FastSurfaceletTransform3D):
        super().__init__()
        self.forward_t: FastSurfaceletTransform3D = forward_transform_instance
        self.dtype_idx: FPDTypeIdx = self.forward_t.dtype_idx

    @override
    def is_valid_input(self, z: torch.Tensor) -> bool:
        return z.dtype.is_complex and (z.ndim == 5)

    @override
    def transform(self, z: torch.Tensor) -> torch.Tensor:
        B, C_total, D_dim, H, W = z.shape
        masks = self.forward_t._get_wrapping_masks(D_dim, H, W, z.device)
        num_masks = len(masks)

        if C_total % num_masks != 0:
            raise ValueError(f"입력 채널({C_total})이 타일 마스크 수({num_masks})와 일치하지 않습니다.")

        C = C_total // num_masks
        z_reshaped = z.view(B, num_masks, C, D_dim, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()

        X_f_recon = torch.zeros(
            (B, C, D_dim, H, W),
            dtype=get_complex_dtype(self.dtype_idx),
            device=z.device,
        )

        for i, mask in enumerate(masks):
            band_spatial = z_reshaped[:, :, i, :, :, :]
            band_f_ishift = torch.fft.fftn(band_spatial, dim=(-3, -2, -1))
            band_f = torch.fft.fftshift(band_f_ishift, dim=(-3, -2, -1))
            X_f_recon += band_f * mask.unsqueeze(0).unsqueeze(0)

        x_recon = torch.fft.ifftn(
            torch.fft.ifftshift(X_f_recon, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
        )

        return x_recon.real

    @override
    def get_inverse_transform(self) -> FastSurfaceletTransform3D:
        return self.forward_t