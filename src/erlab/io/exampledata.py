"""Generates simple simulated ARPES data for testing and demonstration."""

__all__ = [
    "generate_data",
    "generate_data_angles",
    "generate_gold_edge",
    "generate_hvdep_cuts",
]

import typing
from collections.abc import Hashable, Sequence

import numpy as np
import scipy.ndimage
import scipy.special
import xarray as xr

import erlab


def _func(kvec, a):
    n1 = np.array([a / np.sqrt(3), 0])
    n2 = np.array([-a / 2 / np.sqrt(3), a / 2])
    n3 = np.array([-a / 2 / np.sqrt(3), -a / 2])
    return (
        np.exp(1j * np.dot(kvec, n1))
        + np.exp(1j * np.dot(kvec, n2))
        + np.exp(1j * np.dot(kvec, n3))
    )


def _calc_graphene_mat_el(alpha, beta, polarization, factor_f, factor_g):
    eps_m, eps_0, eps_p = np.linalg.solve(
        np.array(
            [
                [1 / np.sqrt(2), -1j / np.sqrt(2), 0],
                [0.0, 0.0, 1.0],
                [-1 / np.sqrt(2), -1j / np.sqrt(2), 0],
            ]
        ).T,
        polarization,
    )

    d_channel = -factor_f * (
        np.sqrt(3 / 10) * eps_m * scipy.special.sph_harm_y(2, 1, beta, alpha)
        - np.sqrt(2 / 5) * eps_0 * scipy.special.sph_harm_y(2, 0, beta, alpha)
        + np.sqrt(3 / 10) * eps_p * scipy.special.sph_harm_y(2, -1, beta, alpha)
    )
    s_channel = factor_g * eps_0 * scipy.special.sph_harm_y(0, 0, beta, alpha)
    return d_channel + s_channel


def _fermi_dirac(E, T):
    return 1 / (np.exp(E / (8.61733e-5 * T)) + 1)


def _band(kvec, t, a):
    return -t * np.abs(_func(kvec, a))


def _spectral_function(w, bareband, Sreal, Simag):
    return Simag / (np.pi * ((w - bareband - Sreal) ** 2 + Simag**2))


def _add_fd_norm(image, eV, temp=30, efermi=0, count=1e7, const_bkg=1e-3) -> None:
    if temp != 0:
        image *= _fermi_dirac(eV - efermi, temp)[None, None, :]
    image *= count
    image += const_bkg * count


# Changing default values may break tests
def generate_data(
    shape: tuple[int, int, int] = (250, 250, 300),
    *,
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
    const_bkg: float = 1e-3,
) -> xr.DataArray:
    """Generate simulated data for a given shape in momentum space.

    Generates simulated ARPES data based on a simple graphene-like tight-binding model.

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
    const_bkg
        Constant background as a fraction of the count, by default 1e-3

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

    Eij = _band(point_iter, t, a).reshape(*shape[:-1], 1)

    Akw_p = _spectral_function(eV[None, None, :], Eij + bandshift, Sreal, Simag)
    Akw_m = _spectral_function(eV[None, None, :], -Eij + bandshift, Sreal, Simag)

    phase = np.angle(_func(point_iter, a)).reshape(shape[0], shape[1])
    Akw_p *= np.abs(1 + np.cos(phase))[:, :, None] ** 2
    Akw_m *= np.abs(1 - np.cos(phase))[:, :, None] ** 2
    out = Akw_p + Akw_m

    _add_fd_norm(out, eV, efermi=0, temp=temp, count=count * 1e-6, const_bkg=const_bkg)

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
    *,
    angrange: float | tuple[float, float] | dict[str, tuple[float, float]] = 15.0,
    Erange: tuple[float, float] = (-0.45, 0.12),
    hv: float = 50.0,
    configuration: (
        erlab.constants.AxesConfiguration | int
    ) = erlab.constants.AxesConfiguration.Type1,
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
    const_bkg: float = 1e-3,
    assign_attributes: bool = False,
    extended: bool = True,
    polarization: Sequence[float] = (0.0, 0.0, 1.0),
    inner_potential: float = 10.0,
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
    const_bkg
        Constant background as a fraction of the count, by default 1e-3
    assign_attributes
        Whether to assign attributes to the generated data, by default `False`
    extended
        Whether to include additional effects such as kz modulation and polarization, by
        default `True`
    polarization
        Photon polarization vector, by default (0.0, 0.0, 1.0). Only used if `extended`
        is `True`
    inner_potential
        Inner potential in eV, by default 10.0. Only used if `extended` is `True`.

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

    eV = np.linspace(*Erange, shape[2])

    # Pad energy range for gaussian kernel
    eV_extended, gaussian_kernel = erlab.analysis.fit.functions.general._gen_kernel(
        eV, float(Eres), pad=5
    )
    Ekin = hv - 4.5 + eV_extended[None, None, :]
    forward_func, _ = erlab.analysis.kspace.get_kconv_func(
        Ekin, configuration=configuration, angle_params={}
    )

    a_mesh, b_mesh = np.meshgrid(alpha, beta, indexing="ij")
    a_mesh, b_mesh = a_mesh[:, :, None], b_mesh[:, :, None]
    kxv, kyv = forward_func(a_mesh, b_mesh)

    # k-point grid
    point_iter = np.stack([kxv, kyv], axis=3)

    # Energy eigenvalues
    Eij = _band(point_iter, t, a)

    # Spectral function
    Akw_p = _spectral_function(eV_extended, Eij + bandshift, Sreal, Simag)
    Akw_m = _spectral_function(eV_extended, -Eij + bandshift, Sreal, Simag)

    # Matrix element phase
    phase = np.angle(_func(point_iter, a))
    mat_el_p = 1 + np.exp(1j * phase)
    mat_el_m = 1 - np.exp(1j * phase)

    if extended:
        dummy_data = xr.DataArray(
            np.empty((*tuple(shape[:-1]), len(eV_extended))),
            dims=["alpha", "beta", "eV"],
            coords={"alpha": alpha, "beta": beta, "eV": eV_extended, "hv": hv},
        )
        dummy_data = dummy_data.expand_dims("hv")

        Ekin = dummy_data.hv - 4.5 + dummy_data.eV
        forward_func, _ = erlab.analysis.kspace.get_kconv_func(
            Ekin, configuration=configuration, angle_params={}
        )
        kx, ky = forward_func(dummy_data.alpha, dummy_data.beta)

        c1, c2 = 143.0, 0.054
        imfp = (c1 / (Ekin**2) + c2 * np.sqrt(Ekin)) * 10

        # Initial out-of-plane momentum
        kz = erlab.analysis.kspace.kz_func(Ekin, inner_potential, kx, ky)

        k_tot = erlab.constants.rel_kconv * np.sqrt(Ekin)

        # Final out-of-plane momentum
        k_perp = erlab.analysis.kspace._kperp_func(k_tot**2, kx, ky)

        # Photon polarization
        pol = np.array(polarization)

        terms = 1j * kx * pol[0] + 1j * ky * pol[1] + (1j * kz - (1 / imfp)) * pol[2]
        terms = terms * (1 / (1j * (k_perp - kz + 0.0) + 1 / imfp))

        # Recast into numpy array
        terms = terms.squeeze("hv").transpose("alpha", "beta", "eV").values

        terms = terms * _calc_graphene_mat_el(
            np.deg2rad(dummy_data.beta.values)[None, :, None],
            np.deg2rad(dummy_data.alpha.values)[:, None, None],
            polarization=pol,
            factor_f=0.0,
            factor_g=1.0,
        )

        mat_el_p = mat_el_p.astype(np.complex128) * terms
        mat_el_m = mat_el_m.astype(np.complex128) * terms

    out = Akw_p * np.abs(mat_el_p) ** 2 + Akw_m * np.abs(mat_el_m) ** 2

    # Apply momentum broadening
    out = scipy.ndimage.gaussian_filter(
        out,
        sigma=[angres / (a[1] - a[0]) if len(a) > 1 else 0 for a in (alpha, beta)]
        + [0],
        truncate=5.0,
    )

    # Multiply by Fermi-Dirac distribution and add constant background
    _add_fd_norm(
        out, eV_extended, efermi=0, temp=temp, count=count * 1e-6, const_bkg=const_bkg
    )

    # Apply energy broadening
    out = np.apply_along_axis(
        lambda m: np.convolve(m, gaussian_kernel, mode="valid"), axis=-1, arr=out
    )

    if noise:
        rng = np.random.default_rng(seed)
        out = scipy.ndimage.gaussian_filter(
            rng.poisson(out).astype(float), ccd_sigma, truncate=10.0, axes=(0, 2)
        )

    out = xr.DataArray(out, coords={"alpha": alpha, "beta": beta, "eV": eV})
    out = out.assign_coords(xi=0.0, delta=0.0, hv=hv)

    match configuration:
        case (
            erlab.constants.AxesConfiguration.Type1DA
            | erlab.constants.AxesConfiguration.Type2DA
        ):
            out = out.assign_coords(chi=0.0)

    if assign_attributes:
        out = out.assign_attrs(
            configuration=int(configuration), sample_temp=temp, sample_workfunction=4.5
        )

    return out.squeeze()


def generate_hvdep_cuts(
    shape: tuple[int, int, int] = (50, 250, 300),
    *,
    angrange: tuple[float, float] = (-15.0, 15.0),
    Erange: tuple[float, float] = (-0.45, 0.12),
    hvrange: tuple[float, float] = (20.0, 69.0),
    configuration: (
        erlab.constants.AxesConfiguration | int
    ) = erlab.constants.AxesConfiguration.Type1,
    temp: float = 20.0,
    a: float = 6.97,
    t: float = 0.43,
    **kwargs,
):
    """
    Generate simulated hv-dependent cuts.

    Parameters
    ----------
    shape
        The shape of the generated data, by default (250, 250, 300)
    hvrange
        Photon energy range in eV.
    angrange
        Angle range in degrees.
    Erange
        Binding energy range in electronvolts.
    configuration
        The experimental configuration, by default Type1DA
    temp
        The temperature in Kelvins for the Fermi-Dirac cutoff. If 0, no cutoff is
        applied, by default 20.0
    a
        Tight binding parameter :math:`a`, by default 6.97
    t
        Tight binding parameter :math:`t`, by default 0.43
    **kwargs
        Additional keyword arguments to pass to `generate_data_angles`.

    """
    hv = np.linspace(*hvrange, shape[0])

    Ekin = hv - 4.5 + 0.0  # Kinetic energy at the Fermi level

    _, inverse_func = erlab.analysis.kspace.get_kconv_func(
        Ekin, configuration=configuration, angle_params={}
    )
    beta = inverse_func(-2 * np.pi / (a * np.sqrt(3)), 2 * np.pi / (a * 3))[1]

    kwargs.setdefault("assign_attributes", True)
    kwargs.setdefault("extended", True)

    out_list = []
    for hv_i, beta_i in zip(hv, beta, strict=True):
        out_list.append(
            generate_data_angles(
                (shape[1], 1, shape[2]),
                angrange={"alpha": angrange, "beta": (beta_i, beta_i)},
                Erange=Erange,
                hv=hv_i,
                configuration=configuration,
                temp=temp,
                a=a,
                t=t,
                **kwargs,
            )
            .expand_dims("hv")
            .assign_coords(beta=("hv", [beta_i]))
        )

    return xr.combine_by_coords(out_list).transpose("alpha", "eV", "hv")


def generate_gold_edge(
    shape: tuple[int, int] = (200, 300),
    *,
    a: float = -0.05,
    b: float = 0.1,
    c: float = 1e-3,
    temp: float = 100.0,
    Eres: float = 1e-2,
    angres: float = 0.1,
    edge_coeffs: Sequence[float] = (0.04, 1e-5, -3e-4),
    background_coeffs: Sequence[float] = (1.0, 0.0, -2e-3),
    count: int = 100000000,
    noise: bool = True,
    seed: int | None = None,
    ccd_sigma: float = 0.6,
) -> xr.DataArray:
    """
    Generate a curved Fermi edge with a linear density of states.

    Parameters
    ----------
    shape
        Shape of the generated data. Default is (200, 300).
    a
        Slope of the linear density of states. Default is -0.05.
    b
        Intercept of the linear density of states. Default is 0.1.
    c
        Constant background. Default is 1e-3.
    temp
        Temperature in Kelvin. Default is 100.0.
    Eres
        Energy resolution in eV. Default is 1e-2.
    angres
        Angular resolution. Default is 0.1.
    edge_coeffs
        Coefficients for the polynomial equation used to calculate the curved Fermi
        level as a function of angle in degrees. Default is (0.04, 1e-5, -3e-4).
    background_coeffs
        Coefficients for the polynomial equation used to calculate the polynomial
        background of the spectrum as a function of angle in degrees. Negative values
        are clipped. Default is (1.0, 0.0, -2e-3).
    count
        Determines the signal-to-noise ratio when `noise` is `True`. Default is 1e+8.
    noise
        Flag indicating whether to add noise to the spectrum. Default is True.
    seed
        Seed for the random number generator for the noise. Default is None.
    ccd_sigma
        Standard deviation of the Gaussian filter applied to the spectrum. Default is
        0.6.
    const_bkg
        Constant background as a fraction of the count, by default 1e-3
    nx
        Number of angle points. Default is 200.
    ny
        Number of energy points. Default is 300.

    Returns
    -------
    data : xarray.DataArray
        Simulated gold edge spectrum.

    """
    alpha = np.linspace(-15, 15, shape[0])
    eV = np.linspace(-1.3, 0.3, shape[1])

    alpha = xr.DataArray(alpha, dims="alpha", coords={"alpha": alpha})
    eV = xr.DataArray(eV, dims="eV", coords={"eV": eV})

    center = np.polynomial.polynomial.polyval(alpha, edge_coeffs)

    data = (b - c + a * eV) / (
        1 + np.exp((1.0 * eV - center) / max(1e-15, temp * erlab.constants.kb_eV))
    ) + c

    background = np.polynomial.polynomial.polyval(alpha, background_coeffs).clip(min=0)

    data = data * background * count * 1e-6

    if noise:
        rng = np.random.default_rng(seed)
        data[:] = rng.poisson(data).astype(float)

    data = erlab.analysis.image.gaussian_filter(
        data,
        sigma=typing.cast(
            "dict[Hashable, float]",
            {
                "eV": Eres / np.sqrt(8 * np.log(2)),
                "alpha": angres / np.sqrt(8 * np.log(2)),
            },
        ),
    )

    if noise:
        data[:] = scipy.ndimage.gaussian_filter(
            rng.poisson(data).astype(float), sigma=ccd_sigma
        )

    return data.assign_attrs(sample_temp=temp)
