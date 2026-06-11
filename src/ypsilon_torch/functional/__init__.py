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
from .robust_location import (
    median_abs_deviation,
    huber_location,
    tukey_biweight_location,
    trimmed_mean,
    winsorized_mean,
    biweight_midvariance,
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

def arsinh(input: torch.Tensor) -> torch.Tensor:
    """
    ArSinh activation:
        f(x) = asinh(x) = log(x + sqrt(1 + x^2))

    Element-wise, shape-preserving, parameter-free.
    Odd, unbounded, log-compressive (≈ x near 0, ≈ sign(x)·log(2|x|) at the
    tails). A gentle alternative to identity/Tanh for heavy-tailed signals.
    Range: R.
    """
    return torch.asinh(input)

def thash_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ThAShGLU(ThASh-Gated Linear Unit):
        split input into (a, b) along *dim*; f = a ⊙ ThASh(b).

    A Gated Linear Unit variant whose gate ThASh(b) = b / sqrt(1 + b^2) is
    bounded to (-1, 1) and saturates more gently than a sigmoid gate. The
    size of *dim* must be even; the output halves that dimension.
    """
    if input.size(dim) % 2 != 0:
        raise ValueError(
            f"thash_glu requires an even size along dim={dim}, "
            f"got {input.size(dim)}"
        )
    a, b = input.chunk(2, dim=dim)
    return a * thash(b)


# ===================================================================
# Complex-valued activations
# ===================================================================

def complex_thash(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    ComplexThASh: magnitude-ThASh with phase preserved.
        f(z) = (z / |z|) · ThASh(|z|) = z / sqrt(1 + |z|^2)

    Squashes the modulus into (0, 1) while leaving arg(z) untouched.
    Parameter-free, shape-preserving. Accepts complex (or real) input.
    """
    mag2 = (z.real.pow(2) + z.imag.pow(2)) if z.is_complex() else z.pow(2)
    return z / torch.sqrt(1 + mag2)

def complex_hglu(z: torch.Tensor, k: float, eps: float = 1e-12) -> torch.Tensor:
    """
    ComplexHGLU: magnitude-HGLU with phase preserved.
        f(z) = (z / |z|) · HGLU_k(|z|),  HGLU_k(r) = (r + sqrt(k + r^2)) / 2

    Maps the modulus through HGLU (range (0, +inf)) while preserving arg(z).
    The eps-smoothed magnitude keeps the z = 0 point well-defined.
    """
    if not (k > 0):
        raise Exception("k must be positive!")
    if z.is_complex():
        mag = torch.sqrt(z.real.pow(2) + z.imag.pow(2) + eps)
    else:
        mag = torch.sqrt(z * z + eps)
    act = (mag + torch.sqrt(k + mag * mag)) / 2
    return z * (act / mag)

def zrelu(z: torch.Tensor) -> torch.Tensor:
    """
    zReLU (Guberman, 2016): keep z iff both Re(z) > 0 and Im(z) > 0,
    otherwise 0.

        f(z) = z   if Re(z) > 0 and Im(z) > 0
             = 0   otherwise
    """
    if not z.is_complex():
        return torch.relu(z)
    mask = (z.real > 0) & (z.imag > 0)
    return z * mask.to(z.dtype)

def crelu(z: torch.Tensor) -> torch.Tensor:
    """
    CReLU (Trabelsi et al., 2018): apply ReLU separately to the real and
    imaginary parts.

        f(z) = ReLU(Re(z)) + i · ReLU(Im(z))
    """
    if not z.is_complex():
        return torch.relu(z)
    return torch.complex(torch.relu(z.real), torch.relu(z.imag))


# ===================================================================
# Stochastic regularization (functional)
# ===================================================================

def gaussian_noise(
    x: torch.Tensor, sigma: float = 0.1, training: bool = True
) -> torch.Tensor:
    """
    Additive Gaussian noise, applied only during training.
        y = x + sigma · eps,   eps ~ N(0, 1)

    Supports real and complex tensors (complex noise is added to both parts
    with per-part standard deviation ``sigma``).
    """
    if (not training) or sigma == 0.0:
        return x
    if x.is_complex():
        noise = torch.randn_like(x.real) + 1j * torch.randn_like(x.imag)
        return x + sigma * noise.to(x.dtype)
    return x + sigma * torch.randn_like(x)

def arsinh_gaussian_noise(
    x: torch.Tensor,
    sigma: float = 0.1,
    c: float = 1.0,
    training: bool = True,
) -> torch.Tensor:
    """
    Scale-aware additive Gaussian noise injected in arsinh space:
        y = c · sinh( asinh(x / c) + sigma · eps ),   eps ~ N(0, 1)

    Because asinh compresses large magnitudes logarithmically, the effective
    perturbation is (multiplicatively) milder on large-magnitude entries and
    near-additive on small ones — a robust, heavy-tail-friendly noise.
    Real-valued inputs only. Applied only during training.
    """
    if (not training) or sigma == 0.0:
        return x
    z = torch.asinh(x / c)
    z = z + sigma * torch.randn_like(z)
    return c * torch.sinh(z)
