"""Feed-forward (FFN) replacement blocks.

- :class:`HourglassFFN` — wide-narrow-wide stacked-bottleneck FFN
  (arXiv:2602.06471 / 2510.01796).
- :class:`MultiHeadFFN` — hardware-independent re-implementation of the
  Flash Multi-Head FFN architecture (arXiv:2512.06989), without the
  Hopper/Triton-specific fused kernel.
"""

from .hourglass import HourglassFFN
from .flash_multi_head_ffn import MultiHeadFFN

__all__ = [
    "HourglassFFN",
    "MultiHeadFFN",
]
