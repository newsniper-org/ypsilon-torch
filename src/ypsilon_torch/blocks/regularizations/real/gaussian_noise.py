import torch
import torch.nn as nn

from ypsilon_torch.functional import gaussian_noise


class GaussianNoise(nn.Module):
    r"""Additive Gaussian noise, active only during training.

    .. math::

        y = x + \sigma \cdot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1)

    A simple stochastic regularizer (a.k.a. additive input/feature noise).
    During evaluation it is the identity. Supports real and complex tensors.

    Parameters
    ----------
    sigma : float
        Standard deviation of the injected noise. Default ``0.1``.
    """

    __constants__ = ["sigma"]

    def __init__(self, sigma: float = 0.1) -> None:
        super().__init__()
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.sigma: float = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gaussian_noise(x, self.sigma, self.training)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}"
