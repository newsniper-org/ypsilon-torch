"""Bilinear MLP blocks for ypsilon-torch.

Combines the weight-interpretable bilinear-MLP formulation of
Pearce et al. (2024) with AFBO's (ICLR 2025) spatial-channel factorized
asymmetric channel-mapping design.
"""

from .afbo import AsymmetricSpatialChannelFactorizedBilinearMLP
from .base import BilinearMLPBase
from .channel_mapping import (
    GroupedCrossChannelMapping,
    OverlappedCycleChannelMapping,
)
from .spatial_modeling import SpatialModeling

__all__ = [
    "BilinearMLPBase",
    "AsymmetricSpatialChannelFactorizedBilinearMLP",
    "GroupedCrossChannelMapping",
    "OverlappedCycleChannelMapping",
    "SpatialModeling",
]
