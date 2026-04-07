"""Generates simple simulated ARPES data for testing and demonstration."""

__all__ = [
    "generate_data",
    "generate_data_angles",
    "generate_data_dirac",
    "generate_gold_edge",
    "generate_hvdep_cuts",
    "make_mesh_pattern",
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


def _dirac_matrix_element(
    kx: np.ndarray,
    ky: np.ndarray,
    k_tot: np.ndarray,
    alpha_coeff: float,
    beta_coeff: float,
    branch_sign: int,
    spin: typing.Literal["integrated", "up", "down"],
) -> np.ndarray:
    """Angular part of the surface-state matrix element."""
    inv_k_tot = np.divide(1.0, k_tot, out=np.zeros_like(k_tot), where=k_tot > 0)
    inv_k_tot_sq = inv_k_tot**2
    k_par_sq = kx**2 + ky**2

    if spin == "integrated":
        out = alpha_coeff**2 * (1 + branch_sign * kx * inv_k_tot)
        out += (
            0.5
            * beta_coeff**2
            * (
                k_par_sq * inv_k_tot_sq
                + branch_sign * (3 * ky**2 - kx**2) * kx * inv_k_tot_sq * inv_k_tot
            )
        )
    else:
        out = 0.5 * alpha_coeff**2
        out += 0.25 * beta_coeff**2 * k_par_sq * inv_k_tot_sq
        interference = (
            alpha_coeff * beta_coeff / np.sqrt(2) * (kx**2 - ky**2) * inv_k_tot_sq
        )
        if spin == "up":
            out += branch_sign * interference
        else:
            out -= branch_sign * interference

    return np.clip(out, 0.0, None)


def _surface_polarization_weight(
    kx: np.ndarray,
    ky: np.ndarray,
    kinetic_energy: np.ndarray,
    polarization: Sequence[float],
    inner_potential: float,
) -> np.ndarray:
    """Free-electron final-state polarization factor."""
    pol = np.asarray(polarization, dtype=float)
    if pol.shape != (3,):
        raise ValueError("polarization must be a length-3 vector")
    pol_norm = np.linalg.norm(pol)
    if pol_norm == 0:
        raise ValueError("polarization vector must be non-zero")
    pol /= pol_norm

    c1, c2 = 143.0, 0.054
    imfp = (c1 / (kinetic_energy**2) + c2 * np.sqrt(kinetic_energy)) * 10
    kz = erlab.analysis.kspace.kz_func(kinetic_energy, inner_potential, kx, ky)
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    k_perp = np.sqrt(np.clip(k_tot**2 - kx**2 - ky**2, a_min=0, a_max=None))

    numerator = 1j * kx * pol[0] + 1j * ky * pol[1] + (1j * kz - 1 / imfp) * pol[2]
    denominator = 1j * (k_perp - kz) + 1 / imfp
    return np.abs(numerator / denominator) ** 2


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

    a_mesh, b_mesh = np.meshgrid(alpha, beta, indexing="ij")
    a_mesh, b_mesh = a_mesh[:, :, None], b_mesh[:, :, None]
    eV = np.linspace(*Erange, shape[2])

    # Pad energy range for gaussian kernel
    eV_extended, gaussian_kernel = erlab.analysis.fit.functions.general._gen_kernel(
        eV, float(Eres), pad=5
    )
    Ekin = hv - 4.5 + eV_extended[None, None, :]

    kxv, kyv = erlab.analysis.kspace.get_kconv_forward(configuration)(
        a_mesh, b_mesh, Ekin
    )

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
        kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
            dummy_data.alpha, dummy_data.beta, Ekin
        )

        c1, c2 = 143.0, 0.054
        imfp = (c1 / (Ekin**2) + c2 * np.sqrt(Ekin)) * 10

        # Initial out-of-plane momentum
        kz = erlab.analysis.kspace.kz_func(Ekin, inner_potential, kx, ky)

        k_tot = erlab.constants.rel_kconv * np.sqrt(Ekin)

        # Final out-of-plane momentum
        k_perp = np.sqrt(np.clip(k_tot**2 - kx**2 - ky**2, a_min=0, a_max=None))

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


def generate_data_dirac(
    shape: tuple[int, int, int] = (500, 60, 500),
    *,
    angrange: float | tuple[float, float] | dict[str, tuple[float, float]] = 15.0,
    Erange: tuple[float, float] = (-0.45, 0.12),
    hv: float = 50.0,
    configuration: (
        erlab.constants.AxesConfiguration | int
    ) = erlab.constants.AxesConfiguration.Type1,
    temp: float = 20.0,
    dirac_velocity: float = 0.3,
    curvature: float = 0.0,
    gap: float = 0.0,
    bandshift: float = 0.0,
    Sreal: float = 0.0,
    Simag: float = 0.03,
    alpha_coeff: float = 1.0,
    beta_coeff: float = 0.6,
    branch: typing.Literal["both", "upper", "lower"] = "both",
    spin: typing.Literal["integrated", "up", "down"] = "integrated",
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
    """Generate simulated topological surface-state ARPES data in angle space.

    The surface-state dispersion follows a minimal Dirac/Rashba Hamiltonian with an
    optional quadratic curvature and mass gap. The matrix-element texture follows the
    spin-integrated and spin-filtered angular factors discussed by Moser for
    spin-orbit-coupled surface states.

    .. versionadded:: 3.21.0

    Parameters
    ----------
    shape
        The shape of the generated data as ``(alpha, beta, eV)``.
    angrange
        Angle range in degrees. Can be a single float, a tuple of floats representing
        the range, or a dictionary with ``alpha`` and ``beta`` keys mapping to tuples
        representing the range for each dimension, by default 15.0.
    Erange
        Binding energy range in electronvolts, by default (-0.45, 0.12).
    hv
        The photon energy in eV. The sample work function is assumed to be 4.5 eV.
    configuration
        The experimental configuration, by default ``Type1``.
    temp
        The temperature in kelvins for the Fermi-Dirac cutoff. If 0, no cutoff is
        applied.
    dirac_velocity
        Linear Dirac or Rashba coefficient in eV A.
    curvature
        Quadratic curvature coefficient in eV A^2.
    gap
        Dirac mass term in eV. The full gap at the Dirac point is ``2 * gap``.
    bandshift
        The rigid energy shift in eV.
    Sreal
        The real part of the self energy.
    Simag
        The imaginary part of the self energy.
    alpha_coeff
        Weight of the out-of-plane ``spz`` orbital contribution in the matrix element.
    beta_coeff
        Weight of the in-plane orbital contribution in the matrix element.
    branch
        Which branch of the surface state to include: ``"upper"``, ``"lower"``, or
        ``"both"``.
    spin
        Matrix-element channel: spin-integrated (default) or one spin-projected
        component (``"up"`` or ``"down"``).
    angres
        Broadening in angle in degrees.
    Eres
        Broadening in energy in electronvolts.
    noise
        Whether to add noise to the generated data.
    seed
        Seed for the random number generator for the noise.
    count
        Determines the signal-to-noise ratio when ``noise`` is ``True``.
    ccd_sigma
        The sigma value for CCD noise generation when ``noise`` is ``True``.
    const_bkg
        Constant background as a fraction of the count.
    assign_attributes
        Whether to assign attributes to the generated data.
    extended
        Whether to include the free-electron final-state polarization prefactor.
    polarization
        Photon polarization vector. Only used if ``extended`` is ``True``.
    inner_potential
        Inner potential in eV. Only used if ``extended`` is ``True``.

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

    if branch not in {"both", "upper", "lower"}:
        raise ValueError("branch must be 'both', 'upper', or 'lower'")
    if spin not in {"integrated", "up", "down"}:
        raise ValueError("spin must be 'integrated', 'up', or 'down'")

    eV = np.linspace(*Erange, shape[2])
    eV_extended, gaussian_kernel = erlab.analysis.fit.functions.general._gen_kernel(
        eV, float(Eres), pad=5
    )
    kinetic_energy = hv - 4.5 + eV_extended[None, None, :]
    if np.any(kinetic_energy <= 0):
        raise ValueError(
            "Photon energy and energy range must give positive photoelectron energy"
        )

    a_mesh, b_mesh = np.meshgrid(alpha, beta, indexing="ij")
    a_mesh, b_mesh = a_mesh[:, :, None], b_mesh[:, :, None]
    kxv, kyv = erlab.analysis.kspace.get_kconv_forward(configuration)(
        a_mesh, b_mesh, kinetic_energy
    )

    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    k_par_sq = kxv**2 + kyv**2
    k_par = np.sqrt(k_par_sq)
    dirac_split = np.sqrt((dirac_velocity * k_par) ** 2 + gap**2)
    energy_upper = curvature * k_par_sq + dirac_split
    energy_lower = curvature * k_par_sq - dirac_split

    spectral_upper = _spectral_function(
        eV_extended, energy_upper + bandshift, Sreal, Simag
    )
    spectral_lower = _spectral_function(
        eV_extended, energy_lower + bandshift, Sreal, Simag
    )

    if extended:
        pol_weight = _surface_polarization_weight(
            kxv, kyv, kinetic_energy, polarization, inner_potential
        )
    else:
        pol_weight = np.ones_like(spectral_upper)

    out = np.zeros_like(spectral_upper)
    if branch in {"both", "upper"}:
        out += spectral_upper * _dirac_matrix_element(
            kxv, kyv, k_tot, alpha_coeff, beta_coeff, +1, spin
        )
    if branch in {"both", "lower"}:
        out += spectral_lower * _dirac_matrix_element(
            kxv, kyv, k_tot, alpha_coeff, beta_coeff, -1, spin
        )
    out *= pol_weight

    out = scipy.ndimage.gaussian_filter(
        out,
        sigma=[
            angres / (axis[1] - axis[0]) if len(axis) > 1 else 0
            for axis in (alpha, beta)
        ]
        + [0],
        truncate=5.0,
    )

    _add_fd_norm(
        out,
        eV_extended,
        efermi=0,
        temp=temp,
        count=count * 1e-6,
        const_bkg=const_bkg,
    )
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
            configuration=int(configuration),
            sample_temp=temp,
            sample_workfunction=4.5,
            dirac_velocity=dirac_velocity,
            curvature=curvature,
            dirac_gap=gap,
            dirac_branch=branch,
            dirac_spin=spin,
            dirac_alpha_coeff=alpha_coeff,
            dirac_beta_coeff=beta_coeff,
        )

    return out.squeeze()


def generate_hvdep_cuts(
    shape: tuple[int, int, int] = (50, 250, 300),
    *,
    angrange: tuple[float, float] = (-15.0, 15.0),
    Erange: tuple[float, float] = (-0.45, 0.12),
    hvrange: tuple[float, float] = (20.0, 69.0),
    hv_shift: float | tuple[float, float] | np.ndarray | None = None,
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
    hv_shift
        Optional energy shift (in eV) applied across the photon-energy axis. If a
        float, a linear shift from ``-hv_shift`` to ``+hv_shift`` is applied across the
        hν range. If a tuple, the shift is linearly interpolated between the two values
        across hν. If an array, it must have length ``shape[0]`` and provides the shift
        at each hν value. If `None`, no shift is applied. The shift is simulated by
        generating each hν cut on a shifted energy axis and then assigning a common
        energy coordinate, so the output contains no all-NaN energies.
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

    beta = erlab.analysis.kspace.get_kconv_inverse(configuration)(
        kx=-2 * np.pi / (a * np.sqrt(3)),
        ky=2 * np.pi / (a * 3),
        kperp=None,
        kinetic_energy=Ekin,
    )[1]

    kwargs.setdefault("assign_attributes", True)
    kwargs.setdefault("extended", True)

    if hv_shift is None:
        shift_vals = np.zeros_like(hv)
    elif isinstance(hv_shift, float):
        shift_vals = np.linspace(-hv_shift, hv_shift, hv.size)
    elif isinstance(hv_shift, tuple):
        shift_vals = np.linspace(hv_shift[0], hv_shift[1], hv.size)
    else:
        shift_vals = np.asarray(hv_shift, dtype=float)
        if shift_vals.shape != hv.shape:
            raise ValueError("hv_shift array must have the same length as the hv axis")

    eV_common = np.linspace(*Erange, shape[2])

    out_list = []
    for hv_i, beta_i, shift_i in zip(hv, beta, shift_vals, strict=True):
        out_list.append(
            generate_data_angles(
                (shape[1], 1, shape[2]),
                angrange={"alpha": angrange, "beta": (beta_i, beta_i)},
                Erange=(Erange[0] - shift_i, Erange[1] - shift_i),
                hv=float(hv_i),
                configuration=configuration,
                temp=temp,
                a=a,
                t=t,
                **kwargs,
            )
            .expand_dims("hv")
            .assign_coords(beta=("hv", [beta_i]), eV=eV_common)
        )

    out = xr.combine_by_coords(out_list).transpose("alpha", "eV", "hv")

    if hv_shift is not None:
        out = out.assign_coords(hv_shift=("hv", shift_vals))

    return out


def _smooth_opening_1d(u, width, sigma):
    """
    Gaussian-broadened 1D top-hat centered at 0 with width ``width``.

    Parameters
    ----------
    u : array_like
        Coordinate (can be ndarray).
    width : float
        Opening width.
    sigma : float
        Gaussian sigma (edge broadening) in same units as u.

    Returns
    -------
    prof : ndarray
        Values between ~0 (bar) and ~1 (open).
    """
    if sigma <= 0:
        # hard top-hat
        return (np.abs(u) <= width / 2).astype(float)

    s2 = np.sqrt(2.0) * sigma
    half = width / 2.0

    # Unnormalized top-hat ⊗ Gaussian
    prof = 0.5 * (
        scipy.special.erf((u + half) / s2) - scipy.special.erf((u - half) / s2)
    )

    center_val = scipy.special.erf(half / s2)
    if center_val > 0:
        prof /= center_val
    return prof


def make_mesh_pattern(
    shape: tuple[int, int],
    *,
    pitch: float = 12.4,
    duty: float = 0.8,
    amplitude: float = 0.3,
    rotate: float = 27.0,
    sigma_edge: float = 0.7,
) -> np.ndarray:
    """Simulate an ARPES mesh pattern using step edges broadened by Gaussians.

    Parameters
    ----------
    shape
        Output image size.
    pitch
        Period of the mesh in pixels.
    duty
        Open fraction along each axis, between 0 and 1.
    amplitude
        Amplitude of the mesh.
    rotate
        Rotation of the mesh relative to detector x-axis.
    sigma_edge
        Gaussian sigma (in pixels) for edge broadening.
    """
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    theta = np.deg2rad(rotate)

    # Rotate into mesh-aligned coordinates
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)

    open_width = duty * pitch

    # Fold coordinates into a single cell centered at 0
    ux = ((xr + pitch / 2) % pitch) - pitch / 2
    uy = ((yr + pitch / 2) % pitch) - pitch / 2

    # 1D opening profiles along each axis (0..1)
    open_x = _smooth_opening_1d(ux, open_width, sigma_edge)
    open_y = _smooth_opening_1d(uy, open_width, sigma_edge)

    mask = 1 - np.minimum(open_x, open_y)
    mask = mask - mask.mean()

    return 1 + amplitude * mask


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
    count: int = 1000000000,
    noise: bool = True,
    seed: int | None = None,
    ccd_sigma: float = 0.6,
    add_mesh: bool = False,
    mesh_params: dict | None = None,
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
    add_mesh
        Whether to multiply the spectrum by a mesh pattern. Default is False.
    mesh_params
        Parameters for the mesh pattern generation. Default is None.

    Returns
    -------
    data : xarray.DataArray
        Simulated gold edge spectrum.

    """
    if add_mesh:
        if mesh_params is None:
            mesh_params = {}
        mesh = make_mesh_pattern(shape, **mesh_params).T
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

    if add_mesh:
        data = data * mesh
    if noise:
        data[:] = scipy.ndimage.gaussian_filter(
            rng.poisson(data).astype(float), sigma=ccd_sigma
        )

    return data.assign_attrs(sample_temp=temp)
