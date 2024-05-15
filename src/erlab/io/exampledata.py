"""Generates simple simulated ARPES data for testing purposes."""

__all__ = ["generate_data", "generate_data_angles"]

from collections.abc import Hashable, Sequence
from typing import cast

import numpy as np
import scipy.ndimage
import xarray as xr

import erlab.analysis.image
import erlab.analysis.kspace
from erlab.constants import kb_eV


def func(kvec, a):
    n1 = np.array([a / np.sqrt(3), 0])
    n2 = np.array([-a / 2 / np.sqrt(3), a / 2])
    n3 = np.array([-a / 2 / np.sqrt(3), -a / 2])
    return (
        np.exp(1j * np.dot(kvec, n1))
        + np.exp(1j * np.dot(kvec, n2))
        + np.exp(1j * np.dot(kvec, n3))
    )


def fermi_dirac(E, T):
    return 1 / (np.exp(E / (8.61733e-5 * T)) + 1)


def band(kvec, t, a):
    return -t * np.abs(func(kvec, a))


def spectral_function(w, bareband, Sreal, Simag):
    return Simag / (np.pi * ((w - bareband - Sreal) ** 2 + Simag**2))


def add_fd_norm(image, eV, temp=30, efermi=0, count=1e7):
    if temp != 0:
        image *= fermi_dirac(eV - efermi, temp)[None, None, :]
    image += 0.1e-2
    image /= image.sum()
    image *= count


def generate_data(
    shape: tuple[int, int, int] = (250, 250, 300),
    krange: float | tuple[float, float] | dict[str, tuple[float, float]] = 0.89,
    Erange: tuple[float, float] = (-0.45, 0.09),
    temp: float = 20.0,
    a: float = 6.97,
    t: float = 0.43,
    bandshift: float = 0.0,
    Sreal: float = 0.0,
    Simag: float = 0.03,
    kres: float = 0.01,
    Eres: float = 2.0e-3,
    noise: bool = True,
    seed: int | None = None,
    count: int = 100000000,
    ccd_sigma: float = 0.6,
) -> xr.DataArray:
    """Generate simulated data for a given shape in momentum space.

    Parameters
    ----------
    shape
        The shape of the generated data, by default (250, 250, 300)
    krange
        Momentum range in inverse angstroms. Can be a single float, a tuple of floats
        representing the range, or a dictionary with ``kx`` and ``ky`` keys mapping to
        tuples representing the range for each dimension, by default 0.89
    Erange
        Binding energy range in electronvolts, by default (-0.45, 0.09)
    temp
        The temperature in Kelvins for the Fermi-Dirac cutoff. If 0, no cutoff is
        applied, by default 20.0
    a
        Tight binding parameter :math:`a`, by default 6.97
    t
        Tight binding parameter :math:`t`, by default 0.43
    bandshift
        The rigid energy shift in eV, by default 0.0
    Sreal
        The real part of the self energy, by default 0.0
    Simag
        The imaginary part of the self energy, by default 0.03
    kres
        Broadening in momentum in inverse angstroms, by default 0.01
    Eres
        Broadening in energy in electronvolts, by default 2.0e-3
    noise
        Whether to add noise to the generated data, by default `True`
    seed
        Seed for the random number generator for the noise. Default is None.
    count
        Determines the signal-to-noise ratio when `noise` is `True`, by default 1e+8
    ccd_sigma
        The sigma value for CCD noise generation when `noise` is `True`, by default 0.6

    Returns
    -------
    xarray.DataArray
        The generated data with coordinates for kx, ky, and eV.

    """
    if isinstance(krange, dict):
        kx = np.linspace(*krange["kx"], shape[0])
        ky = np.linspace(*krange["ky"], shape[1])
    elif isinstance(krange, tuple):
        kx = np.linspace(*krange, shape[0])
        ky = np.linspace(*krange, shape[1])
    else:
        kx = np.linspace(-krange, krange, shape[0])
        ky = np.linspace(-krange, krange, shape[1])

    eV = np.linspace(*Erange, shape[2])

    dE = eV[1] - eV[0]

    point_iter = np.array(np.meshgrid(kx, ky)).T.reshape(-1, 2)

    Eij = band(point_iter, t, a).reshape(*shape[:-1], 1)

    Akw_p = spectral_function(eV[None, None, :], Eij + bandshift, Sreal, Simag)
    Akw_m = spectral_function(eV[None, None, :], -Eij + bandshift, Sreal, Simag)

    phase = np.angle(func(point_iter, a)).reshape(shape[0], shape[1])
    Akw_p *= (1 + np.cos(phase))[:, :, None]
    Akw_m *= (1 - np.cos(phase))[:, :, None]
    out = Akw_p + Akw_m

    add_fd_norm(out, eV, efermi=0, temp=temp, count=count)

    if noise:
        rng = np.random.default_rng(seed)
        out = rng.poisson(out).astype(float)

    broadened = scipy.ndimage.gaussian_filter(
        out,
        sigma=[kres / (k[1] - k[0]) if len(k) > 1 else 0 for k in (kx, ky)]
        + [Eres / dE],
        truncate=10.0,
    )
    if noise:
        broadened = scipy.ndimage.gaussian_filter(
            rng.poisson(broadened).astype(float), ccd_sigma, truncate=10.0
        )

    return xr.DataArray(broadened, coords={"kx": kx, "ky": ky, "eV": eV}).squeeze()


def generate_data_angles(
    shape: tuple[int, int, int] = (500, 60, 500),
    angrange: float | tuple[float, float] | dict[str, tuple[float, float]] = 15.0,
    Erange: tuple[float, float] = (-0.45, 0.12),
    hv: float = 50.0,
    configuration: (
        erlab.analysis.kspace.AxesConfiguration | int
    ) = erlab.analysis.kspace.AxesConfiguration.Type1,
    temp: float = 20.0,
    a: float = 6.97,
    t: float = 0.43,
    bandshift: float = 0.0,
    Sreal: float = 0.0,
    Simag: float = 0.03,
    angres: float = 0.1,
    Eres: float = 10.0e-3,
    noise: bool = True,
    seed: int | None = None,
    count: int = 100000000,
    ccd_sigma: float = 0.6,
    assign_attributes: bool = False,
) -> xr.DataArray:
    """Generate simulated data for a given shape in angle space.

    Parameters
    ----------
    shape
        The shape of the generated data, by default (250, 250, 300)
    angrange
        Angle range in degrees. Can be a single float, a tuple of floats representing
        the range, or a dictionary with ``alpha`` and ``beta`` keys mapping to tuples
        representing the range for each dimension, by default 15.0
    Erange
        Binding energy range in electronvolts, by default (-0.45, 0.12)
    hv
        The photon energy in eV. Note that the sample work function is assumed to be 4.5
        eV, by default 30.0
    configuration
        The experimental configuration, by default Type1DA
    temp
        The temperature in Kelvins for the Fermi-Dirac cutoff. If 0, no cutoff is
        applied, by default 20.0
    a
        Tight binding parameter :math:`a`, by default 6.97
    t
        Tight binding parameter :math:`t`, by default 0.43
    bandshift
        The rigid energy shift in eV, by default 0.0
    Sreal
        The real part of the self energy, by default 0.0
    Simag
        The imaginary part of the self energy, by default 0.03
    angres
        Broadening in angle in degrees, by default 0.01
    Eres
        Broadening in energy in electronvolts, by default 2.0e-3
    noise
        Whether to add noise to the generated data, by default `True`
    seed
        Seed for the random number generator for the noise. Default is None.
    count
        Determines the signal-to-noise ratio when `noise` is `True`, by default 1e+8
    ccd_sigma
        The sigma value for CCD noise generation when `noise` is `True`, by default 0.6
    assign_attributes
        Whether to assign attributes to the generated data, by default `False`

    Returns
    -------
    xarray.DataArray
        The generated data with coordinates for alpha, beta, and eV.

    """
    if isinstance(angrange, dict):
        alpha = np.linspace(*angrange["alpha"], shape[0])
        beta = np.linspace(*angrange["beta"], shape[1])
    elif isinstance(angrange, tuple):
        alpha = np.linspace(*angrange, shape[0])
        beta = np.linspace(*angrange, shape[1])
    else:
        alpha = np.linspace(-angrange, angrange, shape[0])
        beta = np.linspace(-angrange, angrange, shape[1])

    if not isinstance(configuration, erlab.analysis.kspace.AxesConfiguration):
        configuration = erlab.analysis.kspace.AxesConfiguration(configuration)

    eV = np.linspace(*Erange, shape[2])

    Ekin = hv - 4.5 + eV[None, None, :]
    forward_func, _ = erlab.analysis.kspace.get_kconv_func(
        Ekin, configuration=configuration, angle_params={}
    )

    dE = eV[1] - eV[0]

    a_mesh, b_mesh = np.meshgrid(alpha, beta, indexing="ij")
    kxv, kyv = forward_func(a_mesh[:, :, None], b_mesh[:, :, None])
    point_iter = np.stack([kxv, kyv], axis=3)

    Eij = band(point_iter, t, a)

    Akw_p = spectral_function(eV, Eij + bandshift, Sreal, Simag)
    Akw_m = spectral_function(eV, -Eij + bandshift, Sreal, Simag)

    phase = np.angle(func(point_iter, a))
    Akw_p *= 1 + np.cos(phase)
    Akw_m *= 1 - np.cos(phase)
    out = Akw_p + Akw_m

    add_fd_norm(out, eV, efermi=0, temp=temp, count=count)

    if noise:
        rng = np.random.default_rng(seed)
        out = rng.poisson(out).astype(float)

    out = scipy.ndimage.gaussian_filter(
        out,
        sigma=[angres / (a[1] - a[0]) if len(a) > 1 else 0 for a in (alpha, beta)]
        + [Eres / dE],
        truncate=10.0,
    )
    if noise:
        out = scipy.ndimage.gaussian_filter(
            rng.poisson(out).astype(float), ccd_sigma, truncate=10.0, axes=(0, 2)
        )

    out = xr.DataArray(out, coords={"alpha": alpha, "beta": beta, "eV": eV})
    out = out.assign_coords(xi=0.0, delta=0.0, hv=hv)

    match configuration:
        case (
            erlab.analysis.kspace.AxesConfiguration.Type1DA
            | erlab.analysis.kspace.AxesConfiguration.Type2DA
        ):
            out = out.assign_coords(chi=0.0)

    if assign_attributes:
        out = out.assign_attrs(
            configuration=int(configuration),
            temp_sample=temp,
            sample_workfunction=4.5,
        )

    return out.squeeze()


def generate_gold_edge(
    a: float = -0.05,
    b: float = 0.1,
    c: float = 0.0,
    temp: float = 100.0,
    Eres: float = 1e-2,
    angres: float = 0.1,
    edge_coeffs: Sequence[float] = (0.04, 1e-5, -3e-4),
    background_coeffs: Sequence[float] = (1.0, 0.0, -2e-3),
    count: int = 1000000,
    noise: bool = True,
    seed: int | None = None,
    ccd_sigma: float = 0.6,
    nx: int = 200,
    ny: int = 300,
) -> xr.DataArray:
    """
    Generate a curved Fermi edge with a linear density of states.

    Parameters
    ----------
    a
        Slope of the linear density of states. Default is -0.05.
    b
        Intercept of the linear density of states. Default is 0.1.
    c
        Constant background. Default is 0.0.
    temp
        Temperature in Kelvin. Default is 100.0.
    Eres
        Energy resolution in eV. Default is 1e-2.
    angres
        Angular resolution. Default is 0.1.
    edge_coeffs
        Coefficients for the polynomial equation used to calculate the center of the
        spectrum. Default is (0.04, 1e-5, -3e-4).
    background_coeffs
        Coefficients for the polynomial equation used to calculate the background of the
        spectrum. Default is (1.0, 0.0, -2e-3).
    count
        Total count of the spectrum. Default is 1e6.
    noise
        Flag indicating whether to add noise to the spectrum. Default is True.
    seed
        Seed for the random number generator for the noise. Default is None.
    ccd_sigma
        Standard deviation of the Gaussian filter applied to the spectrum. Default is
        0.6.
    nx
        Number of angle points. Default is 200.
    ny
        Number of energy points. Default is 300.

    Returns
    -------
    data : xarray.DataArray
        Simulated gold edge spectrum.

    """
    alpha = np.linspace(-15, 15, nx)
    eV = np.linspace(-1.3, 0.3, ny)

    alpha = xr.DataArray(alpha, dims="alpha", coords={"alpha": alpha})
    eV = xr.DataArray(eV, dims="eV", coords={"eV": eV})

    center = np.polynomial.polynomial.polyval(alpha, edge_coeffs)

    data = (b - c + a * eV) / (
        1 + np.exp((1.0 * eV - center) / max(1e-15, temp * kb_eV))
    ) + c

    background = np.polynomial.polynomial.polyval(alpha, background_coeffs).clip(min=0)

    data *= background
    data += 0.1e-2
    data = data / data.sum()
    data = data * count

    if noise:
        rng = np.random.default_rng(seed)
        data[:] = rng.poisson(data).astype(float)

    data = erlab.analysis.image.gaussian_filter(
        data, sigma=cast(dict[Hashable, float], {"eV": Eres, "alpha": angres})
    )

    if noise:
        data[:] = scipy.ndimage.gaussian_filter(
            rng.poisson(data).astype(float), sigma=ccd_sigma
        )

    return data.assign_attrs(temp_sample=temp)
