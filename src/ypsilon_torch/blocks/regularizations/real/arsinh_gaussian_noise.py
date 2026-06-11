import torch
import torch.nn as nn

from ypsilon_torch.functional import arsinh_gaussian_noise


class ArsinhGaussianNoise(nn.Module):
    r"""Scale-aware Gaussian noise injected in arsinh space (training only).

    .. math::

        y = c \cdot \sinh\!\big( \operatorname{arsinh}(x / c)
                                  + \sigma \cdot \epsilon \big),
        \qquad \epsilon \sim \mathcal{N}(0, 1)

    Because :math:`\operatorname{arsinh}` compresses large magnitudes
    logarithmically, the perturbation acts roughly *multiplicatively* on
    large-magnitude entries and roughly *additively* on small ones. This
    makes it a robust, heavy-tail-friendly alternative to plain additive
    noise. During evaluation it is the identity. Real-valued inputs only.

    Parameters
    ----------
    sigma : float
        Noise standard deviation in arsinh space. Default ``0.1``.
    c : float
        arsinh scale (the magnitude around which the noise transitions from
        additive to multiplicative behaviour). Default ``1.0``.
    """

    __constants__ = ["sigma", "c"]

    def __init__(self, sigma: float = 0.1, c: float = 1.0) -> None:
        super().__init__()
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if c <= 0.0:
            raise ValueError(f"c must be positive, got {c}")
        self.sigma: float = sigma
        self.c: float = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return arsinh_gaussian_noise(x, self.sigma, self.c, self.training)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}, c={self.c}"
