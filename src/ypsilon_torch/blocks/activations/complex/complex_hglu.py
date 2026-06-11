import torch
import torch.nn as nn

try:
    # Python 3.12+
    from typing import override
except Exception:  # pragma: no cover
    from typing_extensions import override

from ypsilon_torch.blocks.activations import ComplexActivationFunctionBase
from ypsilon_torch.functional import complex_hglu


class ComplexHGLU(ComplexActivationFunctionBase):
    r"""
    ComplexHGLU: magnitude-HGLU with phase preserved.

    Definition:
        f(z) = (z / |z|) · HGLU_k(|z|),
        HGLU_k(r) = (r + sqrt(k + r^2)) / 2

    Maps the modulus through HGLU (range (0, +inf)) while preserving arg(z)
    — the complex-domain analogue of the real :class:`HGLU`. An
    epsilon-smoothed magnitude keeps z = 0 well-defined.

    Parameters
    ----------
    k : float
        Positive hyperparameter of HGLU.
    eps : float
        Magnitude-smoothing epsilon. Default ``1e-12``.
    """

    __constants__ = ["k", "eps"]

    @override
    def __init__(self, k: float, eps: float = 1e-12) -> None:
        super().__init__()
        if not (k > 0):
            raise ValueError(f"k must be positive, got {k}")
        self.k: float = k
        self.eps: float = eps

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return complex_hglu(z, self.k, eps=self.eps)

    def extra_repr(self) -> str:
        return f"k={self.k}, eps={self.eps}"
