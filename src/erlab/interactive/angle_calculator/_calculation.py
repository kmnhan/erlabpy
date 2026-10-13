"""Analyzer geometry for the Angle calculator.

Angles are in degrees, energies in eV, and momenta in inverse angstroms.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import erlab.analysis.kspace
import erlab.constants
from erlab.interactive.angle_calculator._low_energy import (
    momentum_to_slopes,
    slopes_to_momentum,
    slopes_to_momentum_at_normal,
)


@dataclasses.dataclass(frozen=True)
class Geometry:
    """Experimental geometry with absolute mechanical angles and no offsets."""

    photon_energy: float = 90.0
    binding_energy: float = 0.0
    analyzer_work_function: float = 4.3
    sample_work_function: float = 4.3
    azimuth: float = 0.0
    polar: float = 0.0
    tilt: float = 0.0
    configuration: int = 3
    low_energy_correction: bool = False
    inner_potential: float = 10.0


def _angle_params(geometry: Geometry) -> dict[str, float]:
    if geometry.configuration in (1, 2):
        return {
            "delta": geometry.azimuth,
            "xi": geometry.tilt if geometry.configuration == 1 else geometry.polar,
        }
    return {"delta": geometry.azimuth, "chi": geometry.polar, "xi": geometry.tilt}


def _validate_correction_geometry(geometry: Geometry) -> None:
    if geometry.low_energy_correction:
        params = _angle_params(geometry)
        if params["xi"] != 0.0 or params.get("chi", 0.0) != 0.0:
            raise ValueError("Low-energy correction requires normal emission geometry.")


def emission_radius(geometry: Geometry, *, include_analyzer: bool = False) -> float:
    """Return the sample escape radius, or NaN below the vacuum threshold.

    With include_analyzer, also require transmission to the analyzer. The smaller
    sample or analyzer kinetic energy then sets the momentum limit.
    """
    work_function = geometry.sample_work_function
    if include_analyzer:
        if not np.isfinite(geometry.analyzer_work_function):
            return float("nan")
        work_function = max(work_function, geometry.analyzer_work_function)
    energy = geometry.photon_energy + geometry.binding_energy - work_function
    if not np.isfinite(energy) or energy < 0.0:
        return float("nan")
    return float(erlab.constants.rel_kconv * np.sqrt(energy))


def angles_to_momentum(alpha, beta, geometry: Geometry, photon_energy=None):
    """Convert broadcast angle arrays; mask inaccessible and backward rays with NaN.

    In Type 1, beta is mechanical polar and xi is tilt. In Type 2, beta is
    mechanical tilt and xi is polar. In DA configurations beta is a deflector angle.
    """
    _validate_correction_geometry(geometry)
    photon_energy = geometry.photon_energy if photon_energy is None else photon_energy
    alpha, beta, photon_energy = np.broadcast_arrays(
        np.asarray(alpha, dtype=float),
        np.asarray(beta, dtype=float),
        np.asarray(photon_energy, dtype=float),
    )
    if (
        geometry.low_energy_correction
        and geometry.configuration in (1, 2)
        and np.any(np.isfinite(beta) & (beta != 0.0))
    ):
        raise ValueError("Low-energy correction requires zero mechanical beta.")
    analyzer_energy = (
        photon_energy + geometry.binding_energy - geometry.analyzer_work_function
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        kx, ky, kperp = erlab.analysis.kspace.get_kconv_forward(geometry.configuration)(
            alpha,
            beta,
            analyzer_energy,
            return_kperp=True,
            **_angle_params(geometry),
        )
        valid = np.isfinite(alpha + beta) & (kperp >= 0.0)
        if geometry.configuration in (3, 4):
            valid &= np.hypot(alpha, beta) <= 90.0
        else:
            valid &= np.abs(alpha) <= 90.0
        valid &= analyzer_energy > 0.0
        if geometry.low_energy_correction:
            # Use the common ray for DA rotation-vector components.
            kx, ky = slopes_to_momentum(
                kx / kperp,
                ky / kperp,
                photon_energy + geometry.binding_energy - geometry.sample_work_function,
                analyzer_energy,
            )
            valid &= kperp > 0.0
    return np.where(valid, kx, np.nan), np.where(valid, ky, np.nan)


def momentum_to_angles(kx, ky, geometry: Geometry):
    """Convert broadcast momenta; return NaN outside the physical emission domain."""
    _validate_correction_geometry(geometry)
    kx, ky = np.broadcast_arrays(
        np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)
    )
    analyzer_energy = (
        geometry.photon_energy
        + geometry.binding_energy
        - geometry.analyzer_work_function
    )
    analyzer_radius = erlab.constants.rel_kconv * np.sqrt(max(analyzer_energy, 0.0))
    transverse = np.hypot(kx, ky)
    valid = (transverse <= analyzer_radius) & (analyzer_radius > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if geometry.low_energy_correction:
            slope_x, slope_y = momentum_to_slopes(
                kx,
                ky,
                geometry.photon_energy
                + geometry.binding_energy
                - geometry.sample_work_function,
                analyzer_energy,
            )
            scale = analyzer_radius / np.sqrt(1.0 + slope_x**2 + slope_y**2)
            kx, ky = scale * slope_x, scale * slope_y
        alpha, beta = erlab.analysis.kspace.get_kconv_inverse(geometry.configuration)(
            kx, ky, None, analyzer_energy, **_angle_params(geometry)
        )
        if geometry.low_energy_correction and geometry.configuration in (1, 2):
            valid &= np.isclose(beta, 0.0, atol=1e-10)
            beta = np.zeros_like(beta)
        # The public inverse selects principal trigonometric branches. At extreme
        # mechanical settings, a branch can point into the sample or map to a
        # different momentum. Keep only the forward-emission solution.
        forward_kx, forward_ky, forward_kperp = erlab.analysis.kspace.get_kconv_forward(
            geometry.configuration
        )(alpha, beta, analyzer_energy, return_kperp=True, **_angle_params(geometry))
        valid &= forward_kperp >= 0.0
        valid &= np.isclose(forward_kx, kx, rtol=1e-9, atol=1e-12)
        valid &= np.isclose(forward_ky, ky, rtol=1e-9, atol=1e-12)
    valid &= np.isfinite(alpha) & np.isfinite(beta)
    if geometry.configuration in (3, 4):
        valid &= np.hypot(alpha, beta) <= 90.0
    return np.where(valid, alpha, np.nan), np.where(valid, beta, np.nan)


def angles_to_momentum3d(alpha, beta, geometry: Geometry, photon_energy=None):
    """Return sample momenta using the shared free-electron final-state model.

    Photon energies can broadcast with both angle arrays. With contact-potential
    correction, the sample vacuum level defines kinetic energy and inner
    potential. Otherwise, use the analyzer energy convention of KspaceTool.
    """
    photon_energy = geometry.photon_energy if photon_energy is None else photon_energy
    kx, ky = angles_to_momentum(alpha, beta, geometry, photon_energy)
    work_function = (
        geometry.sample_work_function
        if geometry.low_energy_correction
        else geometry.analyzer_work_function
    )
    energy = np.asarray(photon_energy) + geometry.binding_energy - work_function
    with np.errstate(invalid="ignore"):
        kz = erlab.analysis.kspace.kz_func(energy, geometry.inner_potential, kx, ky)
    return kx, ky, kz


def momentum_surface(kx_axis, ky_axis, geometry: Geometry):
    """Return the constant-energy final-state hemisphere on an in-plane grid."""
    kx, ky = np.meshgrid(kx_axis, ky_axis)
    work_function = (
        geometry.sample_work_function
        if geometry.low_energy_correction
        else geometry.analyzer_work_function
    )
    energy = geometry.photon_energy + geometry.binding_energy - work_function
    limit = energy
    if geometry.low_energy_correction:
        limit = min(
            limit,
            geometry.photon_energy
            + geometry.binding_energy
            - geometry.analyzer_work_function,
        )
    valid = (limit > 0.0) & (kx**2 + ky**2 <= erlab.constants.rel_kconv**2 * limit)
    with np.errstate(invalid="ignore"):
        kz = erlab.analysis.kspace.kz_func(energy, geometry.inner_potential, kx, ky)
    return np.where(valid[..., None], np.stack((kx, ky, kz), axis=-1), np.nan)


def alpha_kz_surface(alpha_axis, kz_axis, beta, geometry: Geometry):
    """Return sample momenta and photon energies on an alpha versus kz grid.

    The fixed angle ray sets transverse momentum relative to the vacuum normal
    momentum. The private contact-potential model adjusts that ratio when used.
    Points below the vacuum or transmission threshold return NaN.
    """
    _validate_correction_geometry(geometry)
    if (
        geometry.low_energy_correction
        and geometry.configuration in (1, 2)
        and beta != 0
    ):
        raise ValueError("Low-energy correction requires zero mechanical beta.")
    alpha, kz = np.meshgrid(alpha_axis, kz_axis)
    ray_x, ray_y, ray_normal = erlab.analysis.kspace.get_kconv_forward(
        geometry.configuration
    )(alpha, beta, 1.0, return_kperp=True, **_angle_params(geometry))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        normal = erlab.analysis.kspace.kperp_from_kz(kz, geometry.inner_potential)
        slope_x, slope_y = ray_x / ray_normal, ray_y / ray_normal
        if geometry.low_energy_correction:
            kx, ky = slopes_to_momentum_at_normal(
                slope_x,
                slope_y,
                normal,
                geometry.sample_work_function - geometry.analyzer_work_function,
            )
            work_function = geometry.sample_work_function
        else:
            kx, ky = slope_x * normal, slope_y * normal
            work_function = geometry.analyzer_work_function
        photon_energy = erlab.analysis.kspace.hv_func(
            kx, ky, kz, geometry.inner_potential, work_function, geometry.binding_energy
        )
        # Reuse the forward conversion for all emission and transmission limits.
        forward = angles_to_momentum3d(alpha, beta, geometry, photon_energy)
        valid = (
            (ray_normal > 1e-12)
            & (kz >= 0.0)
            & (kz**2 >= geometry.inner_potential / erlab.constants.rel_kzconv)
            & np.isfinite(photon_energy)
            & (photon_energy > 0.0)
            & np.isclose(forward[2], kz, rtol=1e-9, atol=1e-10)
            & np.isclose(forward[0], kx, rtol=1e-9, atol=1e-10)
            & np.isclose(forward[1], ky, rtol=1e-9, atol=1e-10)
        )
    surface = np.stack((kx, ky, kz), axis=-1)
    return np.where(valid[..., None], surface, np.nan), np.where(
        valid, photon_energy, np.nan
    )
