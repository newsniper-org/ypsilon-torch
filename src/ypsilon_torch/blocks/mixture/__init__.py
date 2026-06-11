"""Sparse / conditional-computation ("mixture") blocks.

- :class:`MixtureOfDepths` — token-level dynamic compute (arXiv:2404.02258).
- :class:`MixtureOfDepthsAndExperts` — MoD combined with MoE, integrated or
  staged (arXiv:2404.02258, §4).
- :class:`MixtureOfHiddenDimensions` — conditional activation of hidden-dim
  groups (arXiv:2412.05644).
- :class:`MixtureOfLookupExperts` — embedding-input routed experts that
  re-parameterize to a lookup table at inference (arXiv:2503.15798).
- :class:`AttractorPatchNetwork` — prototype-routed low-rank residual patches
  (arXiv:2602.06993).

Blocks that emit an auxiliary (load-balancing / predictor) loss inherit
:class:`ypsilon_torch.blocks.AuxLossModule` (backed by ``torchutils.auxloss``,
the same mechanism as :class:`~ypsilon_torch.blocks.bilinear_mlp.BilinearMLPBase`).
Sum every contribution across a model with
``AuxLossModule.reg_loss.collect(model)``.
"""

from .mixture_of_depths import MixtureOfDepths
from .mixture_of_depths_experts import MixtureOfDepthsAndExperts
from .mixture_of_hidden_dimensions import MixtureOfHiddenDimensions
from .mixture_of_lookup_experts import MixtureOfLookupExperts
from .attractor_patch_network import AttractorPatchNetwork

__all__ = [
    "MixtureOfDepths",
    "MixtureOfDepthsAndExperts",
    "MixtureOfHiddenDimensions",
    "MixtureOfLookupExperts",
    "AttractorPatchNetwork",
]
