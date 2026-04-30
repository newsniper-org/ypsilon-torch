"""Warped DOST — non-learnable, resolution-invariant, data-adaptive transform.

Place at: ``src/ypsilon_torch/blocks/transforms/real_complex/warped_dost.py``.

Design summary
--------------
* Output channel count is fixed at ``C * n_per_axis ** D`` regardless of input
  resolution. This unblocks resolution-invariant downstream processing.
* Per-axis frequency bin boundaries are placed at equal-power quantiles of
  the calibration data spectrum.
* No ``nn.Parameter`` is registered; calibration writes only into plain
  ``dict`` / ``list`` state. Optimizer interaction is impossible.

Calibration interface
---------------------
* One-shot:  ``t.fit(calibration_x)``
* Streaming: ``f = t.fitter``  (property; fresh fitter each access)
              ``for c in chunks: f.accumulate(c)``
              ``f.finalize()``

The ``fitter`` property pattern supports concurrent calibration sessions
(distinct fitters from the same transform are isolated until each
``finalize()``) and is convenient for distributed setups (each rank
holds its own fitter, optionally all-reducing the per-axis power
accumulator before finalize).

Calibration is required for every spatial shape that will be transformed;
calling :meth:`transform` for an uncalibrated spatial shape raises
``RuntimeError``.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod

import torch

try:
    from typing import override
except ImportError:  # pragma: no cover
    from typing_extensions import override  # type: ignore

from ypsilon_torch import COMPLEX_DTYPES_DICT, FLOAT_DTYPES_DICT, FPDTypeIdx
from ypsilon_torch.blocks.transforms import InvertibleTransformsBase


# ---------------------------------------------------------------------------
# Internal boundary helpers (private)
# ---------------------------------------------------------------------------

def _enforce_strict_monotonic(raw: list[int], N: int) -> list[int]:
    fixed = [raw[0]]
    for v in raw[1:]:
        v = max(v, fixed[-1] + 1)
        fixed.append(v)
    if fixed[-1] > N:
        fixed[-1] = N
        for i in range(len(fixed) - 2, 0, -1):
            if fixed[i] >= fixed[i + 1]:
                fixed[i] = fixed[i + 1] - 1
            else:
                break
    return fixed


def _log_uniform_boundaries(N: int, n_per_axis: int) -> list[int]:
    """Safety-net layout used only when calibration data has zero spectrum.

    Layout (with K = n_per_axis):
      band 0       : [0, 1)
      band 1..K-2  : log-spaced inside [1, N/2)
      band K-1     : [N/2, N)
    """
    if n_per_axis < 3:
        raise ValueError(
            f"n_per_axis must be >= 3 (DC + at least 1 log bin + wrap), got {n_per_axis}"
        )
    if N < 4:
        raise ValueError(
            f"axis length N must be >= 4 to support a non-trivial layout, got {N}"
        )
    half = N // 2
    n_log_bins = n_per_axis - 2
    if n_per_axis > half + 2:
        raise ValueError(
            f"n_per_axis={n_per_axis} too large for axis length N={N} "
            f"(requires N >= 2 * (n_per_axis - 2) = {2 * (n_per_axis - 2)})"
        )

    log_spaced = [
        (math.log2(half)) * k / n_log_bins for k in range(n_log_bins + 1)
    ]
    raw = [0, 1]
    for k in range(1, n_log_bins + 1):
        raw.append(int(round(2 ** log_spaced[k])))
    raw.append(N)
    return _enforce_strict_monotonic(raw, N)


def _quantile_boundaries(power: torch.Tensor, N: int, n_per_axis: int) -> list[int]:
    """Equal-power quantile boundaries on the positive-frequency spectrum."""
    if n_per_axis < 3:
        raise ValueError(f"n_per_axis must be >= 3, got {n_per_axis}")
    if N < 4:
        raise ValueError(f"axis length N must be >= 4, got {N}")
    half = N // 2
    if n_per_axis > half + 2:
        raise ValueError(
            f"n_per_axis={n_per_axis} too large for axis length N={N} "
            f"(requires N >= 2 * (n_per_axis - 2) = {2 * (n_per_axis - 2)})"
        )

    n_log_bins = n_per_axis - 2

    if half <= 1:
        return _log_uniform_boundaries(N, n_per_axis)

    positive_power = power[1:half]
    cum = positive_power.cumsum(0)
    total = cum[-1].item() if cum.numel() > 0 else 0.0
    if total <= 0:
        return _log_uniform_boundaries(N, n_per_axis)

    raw = [0, 1]
    for k in range(1, n_log_bins):
        target = total * k / n_log_bins
        idx = int(torch.searchsorted(
            cum, torch.tensor(target, dtype=cum.dtype, device=cum.device)
        ).item())
        raw.append(min(half, idx + 1))
    raw.append(half)
    raw.append(N)
    return _enforce_strict_monotonic(raw, N)


# ---------------------------------------------------------------------------
# Common base for forward / inverse
# ---------------------------------------------------------------------------

class _WarpedDOSTBase[I: '_WarpedDOSTBase'](InvertibleTransformsBase[I], ABC):
    @abstractmethod
    @override
    def __init__(self, D: int, n_per_axis: int) -> None:
        super().__init__()
        self.D: int = D
        self.n_per_axis: int = n_per_axis
        self.n_bands: int = n_per_axis ** D

        self._fitted_boundaries: dict[tuple[int, ...], list[list[int]]] = {}
        self._cache: dict[
            tuple[tuple[int, ...], torch.device],
            tuple[torch.Tensor, tuple[tuple[slice, ...], ...]],
        ] = {}

    def _fft_dims(self) -> tuple[int, ...]:
        return tuple(range(2, 2 + self.D))

    def _resolve_boundaries(self, spatial_shape: tuple[int, ...]) -> list[list[int]]:
        if spatial_shape not in self._fitted_boundaries:
            raise RuntimeError(
                f"WarpedDOST has not been calibrated for spatial_shape={spatial_shape}. "
                f"Use t.fit(calibration_x) (one-shot) or t.fitter.accumulate(...).finalize() "
                f"(streaming) first."
            )
        return self._fitted_boundaries[spatial_shape]

    def _get_cached_buffers(
        self, spatial_shape: tuple[int, ...], device: torch.device
    ) -> tuple[torch.Tensor, tuple[tuple[slice, ...], ...]]:
        cache_key = (spatial_shape, device)
        if cache_key in self._cache:
            return self._cache[cache_key]

        per_axis_boundaries = self._resolve_boundaries(spatial_shape)
        per_axis_slices = [
            [slice(b[i], b[i + 1]) for i in range(self.n_per_axis)]
            for b in per_axis_boundaries
        ]
        band_slices = list(itertools.product(*per_axis_slices))

        mask = torch.zeros(
            (self.n_bands, *spatial_shape), dtype=torch.bool, device=device
        )
        for b, sl in enumerate(band_slices):
            mask[(b,) + sl] = True

        if len(self._cache) > 32:
            self._cache.pop(next(iter(self._cache)))

        result = (mask, tuple(band_slices))
        self._cache[cache_key] = result
        return result

    def _invalidate_cache(self, spatial_shape: tuple[int, ...] | None = None) -> None:
        if spatial_shape is None:
            self._cache.clear()
        else:
            for k in [k for k in self._cache if k[0] == spatial_shape]:
                self._cache.pop(k)

    def is_calibrated_for(self, spatial_shape: tuple[int, ...]) -> bool:
        return spatial_shape in self._fitted_boundaries


# ---------------------------------------------------------------------------
# Fitter — encapsulates one streaming-calibration session
# ---------------------------------------------------------------------------

class WarpedDOSTFitter:
    """One streaming calibration session for a :class:`WarpedDOST`.

    Constructed via the ``transform.fitter`` property. Holds per-axis
    power-sum accumulators internally (``sum_d N_d`` floats, regardless of
    chunk count or chunk size). The owning transform shares no state with
    the fitter until :meth:`finalize` commits boundaries.

    Multiple concurrent fitters from the same transform are isolated;
    each fitter's spatial shape and device are locked on the first
    :meth:`accumulate` call.
    """

    def __init__(self, transform: "WarpedDOST") -> None:
        self._transform: WarpedDOST = transform
        self._spatial_shape: tuple[int, ...] | None = None
        self._device: torch.device | None = None
        self._axis_power_sum: list[torch.Tensor] | None = None
        self._finalized: bool = False

    @torch.no_grad()
    def accumulate(self, chunk: torch.Tensor) -> "WarpedDOSTFitter":
        """Add one chunk to the running statistics. Returns ``self`` for chaining.

        ``chunk`` shape: ``(B, C, *spatial)``, real. The first call locks
        ``spatial`` and the chunk's device for the rest of this fitter's
        lifetime.
        """
        if self._finalized:
            raise RuntimeError("fitter has already been finalized; create a new one")
        if chunk.is_complex():
            raise ValueError("calibration chunk must be real-valued")
        D = self._transform.D
        if chunk.ndim < D + 2:
            raise ValueError(
                f"calibration chunk must have at least {D + 2} dims, "
                f"got shape {tuple(chunk.shape)}"
            )

        spatial_shape = tuple(chunk.shape[2 : 2 + D])
        device = chunk.device

        if self._axis_power_sum is None:
            self._spatial_shape = spatial_shape
            self._device = device
            self._axis_power_sum = [
                torch.zeros(N, dtype=torch.float32, device=device)
                for N in spatial_shape
            ]
        else:
            if spatial_shape != self._spatial_shape:
                raise ValueError(
                    f"chunk spatial shape {spatial_shape} does not match "
                    f"this fitter's locked shape {self._spatial_shape}"
                )
            if device != self._device:
                raise ValueError(
                    f"chunk device {device} does not match this fitter's "
                    f"locked device {self._device}"
                )

        dims = self._transform._fft_dims()
        x_c = self._transform._to_complex(chunk)
        x_freq = torch.fft.fftn(x_c, dim=dims)
        power_full = x_freq.abs().pow(2)

        for d in range(D):
            other_dims = tuple(i for i in range(power_full.ndim) if i != 2 + d)
            chunk_sum = power_full.sum(dim=other_dims).to(torch.float32)
            self._axis_power_sum[d] = self._axis_power_sum[d] + chunk_sum

        return self

    @torch.no_grad()
    def finalize(self) -> None:
        """Compute boundaries from accumulated statistics and commit to transform.

        After this call the fitter is unusable (further ``accumulate`` /
        ``finalize`` calls raise). The transform's ``_fitted_boundaries``
        is updated for the locked spatial shape and the corresponding
        mask cache is invalidated.
        """
        if self._finalized:
            raise RuntimeError("fitter has already been finalized")
        if self._axis_power_sum is None:
            raise RuntimeError(
                "fitter has no accumulated chunks; call accumulate() first"
            )

        per_axis: list[list[int]] = []
        for d, N in enumerate(self._spatial_shape):  # type: ignore[arg-type]
            per_axis.append(
                _quantile_boundaries(
                    self._axis_power_sum[d], N, self._transform.n_per_axis
                )
            )

        self._transform._commit_boundaries(self._spatial_shape, per_axis)  # type: ignore[arg-type]
        self._finalized = True
        self._axis_power_sum = None

    # ---- introspection ----------------------------------------------------

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    @property
    def spatial_shape(self) -> tuple[int, ...] | None:
        return self._spatial_shape

    @property
    def device(self) -> torch.device | None:
        return self._device


# ---------------------------------------------------------------------------
# Forward transform
# ---------------------------------------------------------------------------

class WarpedDOST(_WarpedDOSTBase['InverseWarpedDOST']):
    """Warped DOST with fixed channel expansion ``C * n_per_axis ** D``.

    Calibrate before transforming via either:
      * ``t.fit(calibration_x)`` — one-shot.
      * ``t.fitter`` (property)  — start a streaming session, then
        ``f.accumulate(...)`` repeatedly and ``f.finalize()``.
    """

    @override
    def __init__(self, D: int, n_per_axis: int) -> None:
        super().__init__(D, n_per_axis)

    @override
    def is_valid_input(self, x: torch.Tensor) -> bool:
        return (not x.dtype.is_complex) and (x.ndim >= self.D + 2)

    def _to_complex(
        self, x: torch.Tensor, dtype_idx: FPDTypeIdx | None = None
    ) -> torch.Tensor:
        if dtype_idx is None:
            if x.dtype == torch.float64:
                return x.to(torch.complex128)
            if x.dtype == torch.float16:
                return x.to(torch.complex32)
            return x.to(torch.float32).to(torch.complex64)
        return x.to(FLOAT_DTYPES_DICT[dtype_idx]).to(COMPLEX_DTYPES_DICT[dtype_idx])

    # ---- calibration API --------------------------------------------------

    @property
    def fitter(self) -> WarpedDOSTFitter:
        """Begin a new streaming-calibration session.

        Each access returns a fresh fitter; concurrent sessions are
        isolated until each ``finalize()``.
        """
        return WarpedDOSTFitter(self)

    def _commit_boundaries(
        self,
        spatial_shape: tuple[int, ...],
        per_axis: list[list[int]],
    ) -> None:
        """Install computed boundaries and invalidate stale mask caches.

        Called by :meth:`WarpedDOSTFitter.finalize` and by :meth:`fit`.
        """
        self._fitted_boundaries[spatial_shape] = per_axis
        self._invalidate_cache(spatial_shape)

    @torch.no_grad()
    def fit(self, calibration_x: torch.Tensor) -> None:
        """One-shot calibration.

        Equivalent to ``t.fitter.accumulate(calibration_x).finalize()``.
        Use the ``fitter`` property directly when calibration data does
        not fit in memory at once.
        """
        self.fitter.accumulate(calibration_x).finalize()

    # ---- transform --------------------------------------------------------

    @override
    def transform(
        self, x: torch.Tensor, dtype_idx: FPDTypeIdx | None = None
    ) -> torch.Tensor:
        spatial_shape = tuple(x.shape[2 : 2 + self.D])
        # _resolve_boundaries (called inside _get_cached_buffers) raises if not fitted.
        dims = self._fft_dims()

        x_c = self._to_complex(x, dtype_idx)
        f_x = torch.fft.fftn(x_c, dim=dims).unsqueeze(2)

        mask, _ = self._get_cached_buffers(spatial_shape, f_x.device)
        band_freq = f_x * mask.unsqueeze(0).unsqueeze(0)
        band_time = torch.fft.ifftn(band_freq, dim=tuple(d + 1 for d in dims))

        B, C = x.shape[:2]
        return band_time.reshape(B, C * self.n_bands, *spatial_shape)

    @override
    def get_inverse_transform(self) -> 'InverseWarpedDOST':
        inv = InverseWarpedDOST(self.D, self.n_per_axis)
        inv._fitted_boundaries = {k: list(v) for k, v in self._fitted_boundaries.items()}
        return inv


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------

class InverseWarpedDOST(_WarpedDOSTBase[WarpedDOST]):
    """Inverse of :class:`WarpedDOST`.

    Boundaries must come from a calibrated forward transform — either
    via :meth:`WarpedDOST.get_inverse_transform` or via
    :meth:`share_boundaries_from`.
    """

    @override
    def __init__(self, D: int, n_per_axis: int) -> None:
        super().__init__(D, n_per_axis)

    @override
    def is_valid_input(self, z: torch.Tensor) -> bool:
        return z.dtype.is_complex and (z.ndim >= self.D + 2)

    def share_boundaries_from(self, forward: WarpedDOST) -> None:
        self._fitted_boundaries = {
            k: list(v) for k, v in forward._fitted_boundaries.items()
        }
        self._invalidate_cache()

    @override
    def transform(
        self, z: torch.Tensor, dtype_idx: FPDTypeIdx | None = None
    ) -> torch.Tensor:
        spatial_shape = tuple(z.shape[2 : 2 + self.D])
        dims = self._fft_dims()
        mask, band_slices = self._get_cached_buffers(spatial_shape, z.device)

        B, C_expanded = z.shape[:2]
        if C_expanded % self.n_bands != 0:
            raise RuntimeError(
                f"input channels {C_expanded} not divisible by n_bands={self.n_bands}"
            )
        C = C_expanded // self.n_bands
        z = z.view(B, C, self.n_bands, *spatial_shape)

        f_recon = torch.zeros((B, C, *spatial_shape), device=z.device, dtype=z.dtype)
        for b, sl in enumerate(band_slices):
            zb = z[:, :, b]
            zb_f = torch.fft.fftn(zb, dim=dims)
            f_recon[(...,) + sl] += zb_f[(...,) + sl]
        recon = torch.fft.ifftn(f_recon, dim=dims)

        if dtype_idx is None:
            return recon.real
        return recon.real.to(FLOAT_DTYPES_DICT[dtype_idx])

    @override
    def get_inverse_transform(self) -> WarpedDOST:
        fwd = WarpedDOST(self.D, self.n_per_axis)
        fwd._fitted_boundaries = {
            k: list(v) for k, v in self._fitted_boundaries.items()
        }
        return fwd
