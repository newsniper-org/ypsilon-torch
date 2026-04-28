import torch
import torch.nn as nn
import torch.nn.functional as F

from .robust_stats import (
    frechet_median,
    frechet_median_lp,
    frechet_medoid,
    frechet_medoid_lp,
    sinh_mean_arsinh,
    sinh_rms_arsinh,
    sinh_frechet_median_arsinh,
    sinh_frechet_median_lp_arsinh,
    sinh_frechet_medoid_arsinh,
    sinh_frechet_medoid_lp_arsinh,
    root_frechet_median_square,
    root_frechet_median_lp_square,
    root_frechet_medoid_square,
    root_frechet_medoid_lp_square,
)

def complex_dropout(z: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"dropout probability must be in [0, 1], got {p}")
    if (not training) or p == 0.0:
        return z
    if p == 1.0:
        return torch.zeros_like(z)
    # real-valued mask with the same complex-element shape
    if z.is_complex():
        mask: torch.Tensor = F.dropout(torch.ones_like(z.real), p=p, training=training)
    else:
        mask: torch.Tensor = F.dropout(torch.ones_like(z), p=p, training=training)
    return z * mask

def hglu(input: torch.Tensor, k: float) -> torch.Tensor:
    """
    HGLU_k(Hyperbolic Gain Linear Unit with positive hyperparameter k) activation:
        f(x) = (x + sqrt(k + x^2)) / 2

    Element-wise, shape-preserving, parameter-free.
    Range: (0, +inf) for all real inputs.
    """
    if not (k > 0):
        raise Exception("k must be positive!")
    return (input + torch.sqrt(k + input * input)) / 2

def thash(input: torch.Tensor) -> torch.Tensor:
    """
    ThASh(TanhArSinh) activation:
        f(x) = tanh(asinh(x)) = x / sqrt(1 + x^2)

    Element-wise, shape-preserving, parameter-free.
    Range: (-1, 1) for all real inputs.
    """
    return input / torch.sqrt(1 + input * input)
