"""Private position-limit model for a normal parallel-plate contact potential.

The sample normal must be parallel to the analyzer axis. Slopes are transverse
displacement divided by plate spacing,
not independent tangents of deflector rotation-vector components. Energies are
kinetic energies in eV relative to the respective vacuum levels. Momenta are in
inverse angstroms. Both endpoints must have positive kinetic energy and forward
normal velocity. Invalid points return NaN.
"""

from __future__ import annotations

import numpy as np

import erlab.constants


def slopes_to_momentum(slope_x, slope_y, sample_energy, analyzer_energy):
    """Return conserved transverse momentum from the position-limit ray slopes."""
    slope_x, slope_y, sample_energy, analyzer_energy = np.broadcast_arrays(
        np.asarray(slope_x, dtype=float),
        np.asarray(slope_y, dtype=float),
        np.asarray(sample_energy, dtype=float),
        np.asarray(analyzer_energy, dtype=float),
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        g = (analyzer_energy - sample_energy) / (2.0 * sample_energy)
        q2 = slope_x**2 + slope_y**2
        discriminant = 1.0 + 2.0 * g - g**2 * q2
        fp = np.sqrt((1.0 + g + np.sqrt(discriminant)) / (2.0 * (1.0 + q2)))
        normal_sample = fp - g / (2.0 * fp)
        normal_analyzer = fp + g / (2.0 * fp)
        scale = erlab.constants.rel_kconv * np.sqrt(sample_energy)
        kx, ky = scale * fp * slope_x, scale * fp * slope_y
        radius = erlab.constants.rel_kconv * np.sqrt(
            np.minimum(sample_energy, analyzer_energy)
        )
        valid = (
            np.isfinite(sample_energy)
            & (sample_energy > 0.0)
            & np.isfinite(analyzer_energy)
            & (analyzer_energy > 0.0)
            & np.isfinite(slope_x)
            & np.isfinite(slope_y)
            & (normal_sample >= -1e-12)
            & (normal_analyzer >= -1e-12)
            & (np.hypot(kx, ky) <= radius * (1.0 + 1e-12))
        )
    return np.where(valid, kx, np.nan), np.where(valid, ky, np.nan)


def momentum_to_slopes(kx, ky, sample_energy, analyzer_energy):
    """Return position-limit slopes for forward emission and transmission."""
    kx, ky, sample_energy, analyzer_energy = np.broadcast_arrays(
        np.asarray(kx, dtype=float),
        np.asarray(ky, dtype=float),
        np.asarray(sample_energy, dtype=float),
        np.asarray(analyzer_energy, dtype=float),
    )
    transverse_squared = kx**2 + ky**2
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        sample_normal = np.sqrt(
            erlab.constants.rel_kconv**2 * sample_energy - transverse_squared
        )
        analyzer_normal = np.sqrt(
            erlab.constants.rel_kconv**2 * analyzer_energy - transverse_squared
        )
        normal_sum = sample_normal + analyzer_normal
        slope_x, slope_y = 2.0 * kx / normal_sum, 2.0 * ky / normal_sum
    valid = (
        np.isfinite(slope_x)
        & np.isfinite(slope_y)
        & np.isfinite(sample_energy)
        & (sample_energy > 0.0)
        & np.isfinite(analyzer_energy)
        & (analyzer_energy > 0.0)
    )
    return np.where(valid, slope_x, np.nan), np.where(valid, slope_y, np.nan)


def slopes_to_momentum_at_normal(slope_x, slope_y, sample_normal, energy_difference):
    """Invert a position ray at a fixed sample normal momentum.

    ``energy_difference`` is analyzer minus sample kinetic energy. The plate
    potential changes only the normal kinetic energy. The displacement slope
    uses the mean of the normal momenta at the two plates.
    """
    slope_x, slope_y, sample_normal = np.broadcast_arrays(
        np.asarray(slope_x, dtype=float),
        np.asarray(slope_y, dtype=float),
        np.asarray(sample_normal, dtype=float),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        analyzer_normal = np.sqrt(
            sample_normal**2 + erlab.constants.rel_kconv**2 * energy_difference
        )
        scale = (sample_normal + analyzer_normal) / 2.0
        kx, ky = slope_x * scale, slope_y * scale
    valid = (sample_normal >= 0.0) & np.isfinite(kx) & np.isfinite(ky)
    return np.where(valid, kx, np.nan), np.where(valid, ky, np.nan)
