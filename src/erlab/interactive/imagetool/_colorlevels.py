"""Numerical rules for ImageTool image values and automatic color limits."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class ImageLevels:
    """Resolved automatic limits and whether the display fallback was required."""

    limits: npt.NDArray
    fallback: bool


def image_display_dtype(dtype: np.dtype) -> np.dtype:
    """Keep float32 images and convert all other inputs to float64."""
    return np.dtype(np.float32 if dtype == np.dtype(np.float32) else np.float64)


def limits_require_display_mask(mn: float, mx: float, limit: float) -> bool:
    return bool(np.isinf(mn) or np.isinf(mx) or mn < -limit or mx > limit)


def display_safe_values(values, limit: float):
    """Mask unsafe rendered values without changing a safe or source array."""
    arr = np.asarray(values)
    if (
        arr.size == 0
        or not np.issubdtype(arr.dtype, np.number)
        or np.issubdtype(arr.dtype, np.complexfloating)
    ):
        return values
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not limits_require_display_mask(mn, mx, limit):
        return values
    with np.errstate(all="ignore"):
        return np.where(np.isfinite(arr) & (np.abs(arr) <= limit), arr, np.nan)


def finite_percentile_limits(values: npt.NDArray) -> npt.NDArray | None:
    """Find both percentiles in a private finite buffer, preserving scalar dtype."""
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        return None
    # Match scalar percentile promotion on NumPy 1.x and 2.x.
    q_dtype = np.result_type(finite_values.dtype.type(0), 0.0)
    return np.percentile(
        finite_values,
        np.array([2, 98], dtype=q_dtype),
        overwrite_input=True,
    )


def resolve_image_levels(values: npt.NDArray, *, robust: bool) -> ImageLevels:
    """Resolve limits from already prepared image values using one fallback rule."""
    if robust:
        levels = finite_percentile_limits(values)
    else:
        finite_values = values[np.isfinite(values)]
        levels = (
            np.array([finite_values.min(), finite_values.max()])
            if finite_values.size
            else None
        )
    if levels is None or levels[0] == levels[1]:
        return ImageLevels(limits=np.array([0.0, 255.0]), fallback=True)
    return ImageLevels(limits=levels, fallback=False)
