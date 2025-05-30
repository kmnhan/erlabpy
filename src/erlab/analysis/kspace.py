"""Momentum conversion functions.

Typically, the user will not have to call this module directly, but will instead use the
accessor method :meth:`xarray.DataArray.kspace.convert`.

For more front-end utilities related to momentum conversion, see the documentation of
the :attr:`xarray.DataArray.kspace` accessor.

Angle conventions and function forms are based on Ref. :cite:p:`ishida2018kconv`.

"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import xarray

import erlab.constants
from erlab.constants import AxesConfiguration


def _sinc(x):
    return np.sin(x) / (x + 1e-15)


def _kperp_func(k_tot_sq, kx, ky):
    r""":math:`\sqrt{k^2 - k_x^2 - k_y^2}`."""
    return np.sqrt(np.clip(k_tot_sq - kx**2 - ky**2, a_min=0, a_max=None))


def change_configuration(
    darr: xarray.DataArray, configuration: AxesConfiguration | int
) -> xarray.DataArray:
    """Apply a new axis configuration to the ARPES data.

    Returns a copy of the input data with the coordinates renamed to match the given
    configuration. The original data is not modified.

    This function is useful for setups that are capable of changing the experimental
    geometry.

    Parameters
    ----------
    darr
        The DataArray containing the ARPES data in angle space to modify.
    configuration
        The new configuration to apply.

    Returns
    -------
    xarray.DataArray
        The ARPES data with the new configuration.

    """
    current: AxesConfiguration = darr.kspace.configuration
    new: AxesConfiguration = AxesConfiguration(configuration)

    if current == new:
        return darr.copy()

    # Coord names for each configuration in order of (polar, tilt, deflector)
    coord_order = {
        AxesConfiguration.Type1: ("beta", "xi", "beta_deflector"),
        AxesConfiguration.Type2: ("xi", "beta", "beta_deflector"),
        AxesConfiguration.Type1DA: ("chi", "xi", "beta"),
        AxesConfiguration.Type2DA: ("chi", "xi", "beta"),
    }

    coord_mapping = dict(zip(coord_order[current], coord_order[new], strict=True))

    return (
        darr.copy()
        .rename({k: v for k, v in coord_mapping.items() if k in darr.coords})
        .assign_attrs(configuration=int(new))
    )


def kz_func(kinetic_energy, inner_potential, kx, ky):
    r"""Calculate the out-of-plane momentum inside the sample :math:`k_z`.

    :math:`k_z` is computed from the given kinetic energy :math:`E_k`, inner potential
    :math:`V_0`, and in-plane momenta :math:`k_x`, and :math:`k_y` by

    .. math::

        k_z = \sqrt{k^2 - k_x^2 - k_y^2 + \frac{2 m_e V_0}{\hbar^2}}

    where :math:`k =\sqrt{2 m_e E_k}/\hbar`.
    """
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    k_perp_sq = k_tot**2 - kx**2 - ky**2
    k_z_sq = k_perp_sq + inner_potential / erlab.constants.rel_kzconv
    return np.sqrt(np.clip(k_z_sq, a_min=0, a_max=None))


def kperp_from_kz(kz, inner_potential):
    r"""Calculate the out-of-plane momentum outside the sample :math:`k_\perp`.

    :math:`k_\perp` is computed from the out-of-plane momentum :math:`k_z` and inner
    potential :math:`V_0` by

    .. math::

        k_\perp = \sqrt{k_z^2 - \frac{2 m_e V_0}{\hbar^2}}

    """
    k_perp_sq = kz**2 - inner_potential / erlab.constants.rel_kzconv
    return np.sqrt(np.clip(k_perp_sq, a_min=0, a_max=None))


def hv_func(kx, ky, kz, inner_potential, work_function, binding_energy):
    r"""Calculate the photon energy :math:`hν`.

    The kinetic energy :math:`E_k` is computed from the given out-of-plane momentum
    :math:`k_z`, inner potential :math:`V_0`, and in-plane momenta :math:`k_x`, and
    :math:`k_y` by

    .. math::

        E_k = \frac{\hbar^2}{2 m_e} \left(k_x^2 + k_y^2 + k_z^2\right) - V_0

    Then, the kinetic energy is converted to the photon energy :math:`hν` by

    .. math::

        hν = E_k + \Phi - E_b

    where :math:`\Phi` is the work function of the system and :math:`E_b` is the binding
    energy (negative for occupied states).

    """
    return (
        erlab.constants.rel_kzconv * (kx**2 + ky**2 + kz**2)
        - inner_potential
        + work_function
        - binding_energy
    )


def get_kconv_func(
    kinetic_energy: float | npt.NDArray | xarray.DataArray,
    configuration: AxesConfiguration | int,
    angle_params: dict[str, float],
) -> tuple[Callable, Callable]:
    r"""Return appropriate momentum conversion functions.

    The appropriate function is chosen based on the configuration and kinetic energy.

    Parameters
    ----------
    kinetic_energy
        The kinetic energy of the photoelectrons in eV.
    configuration
        Experimental configuration.
    angle_params
        Dictionary of required angle parameters. If the configuration has a DA, the
        parameters should be `delta`, `chi`, `chi0`, `xi`, and `xi0`. Otherwise, they
        should be `delta`, `xi`, `xi0`, and `beta0`, following the notation in Ref.
        :cite:p:`ishida2018kconv`.

    Returns
    -------
    forward_func : Callable
        Forward function that takes :math:`(α, β)` and returns :math:`(k_x, k_y)`.
    inverse_func : Callable
        Inverse function that takes :math:`(k_x, k_y)` or :math:`(k_x, k_y, k_\perp)`
        and returns :math:`(α, β)`. If :math:`k_\perp` is given, it will return the
        angles broadcasted to :math:`k_\perp` instead of the provided kinetic energy.

    Raises
    ------
    ValueError
        If the given configuration is not valid.

    Note
    ----
    - The only requirement for the input parameters of the returned functions is that
      the shape of the input angles must be broadcastable with each other, and with the
      shape of the kinetic energy array. This means that the shape of the output array
      can be controlled By adjusting the shape of the input arrays. For instance, if the
      kinetic energy is given as a (L, 1, 1) array, :math:`k_x` as a (1, M, 1) array,
      and :math:`k_y` as a (1, 1, N) array, the output angle arrays :math:`α` and
      :math:`β` will each be broadcasted to a (L, M, N) array which can be directly used
      for interpolation.
    - However, the user will not have to worry about the shape of the input arrays,
      because using `xarray.DataArray` objects as the input will most likely broadcast
      the arrays automatically!

    See Also
    --------
    `NumPy Broadcasting Documentation
    <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_

    """
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)

    match configuration:
        case AxesConfiguration.Type1:
            func: Callable = _kconv_func_type1
        case AxesConfiguration.Type2:
            func = _kconv_func_type2
        case AxesConfiguration.Type1DA:
            func = _kconv_func_type1_da
        case AxesConfiguration.Type2DA:
            func = _kconv_func_type2_da
        case _:
            raise ValueError(f"Invalid configuration {configuration}")

    return func(k_tot, **angle_params)


def _kconv_func_type1(k_tot, delta=0.0, xi=0.0, xi0=0.0, beta0=0.0):
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))  # δ
    cx, sx = np.cos(np.deg2rad(xi - xi0)), np.sin(np.deg2rad(xi - xi0))  # ξ - ξ0

    k_tot_sq = k_tot**2

    def _forward_func(alpha, beta):
        alpha_r, beta_r = np.deg2rad(alpha), np.deg2rad(beta - beta0)

        ca, cb = np.cos(alpha_r), np.cos(beta_r)
        sa, sb = np.sin(alpha_r), np.sin(beta_r)

        kx = k_tot * ((sd * sb + cd * sx * cb) * ca - cd * cx * sa)
        ky = k_tot * ((-cd * sb + sd * sx * cb) * ca - sd * cx * sa)

        return kx, ky

    def _inverse_func(kx, ky, kperp=None):
        # cx, sx = np.cos(np.deg2rad(xi)), np.sin(np.deg2rad(xi))  # ξ
        # mask = kx**2 + ky**2 > k_tot_sq + 1e-15
        # kx, ky = np.ma.masked_where(mask, kx), np.ma.masked_where(mask, ky)
        if kperp is None:
            k_sq = k_tot_sq
            k = k_tot
        else:
            k_sq = kx**2 + ky**2 + kperp**2
            k = np.sqrt(k_sq)  # total momentum inside sample

        kperp = _kperp_func(k_sq, kx, ky)

        alpha = np.arcsin((sx * kperp - cx * (cd * kx + sd * ky)) / k)
        beta = np.arctan((sd * kx - cd * ky) / (sx * (cd * kx + sd * ky) + cx * kperp))

        return np.rad2deg(alpha), np.rad2deg(beta) + beta0

    return _forward_func, _inverse_func


def _kconv_func_type2(k_tot, delta=0.0, xi=0.0, xi0=0.0, beta0=0.0):
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))  # δ
    cx, sx = np.cos(np.deg2rad(xi - xi0)), np.sin(np.deg2rad(xi - xi0))  # ξ - ξ0

    k_tot_sq = k_tot**2

    def _forward_func(alpha, beta):
        alpha_r, beta_r = np.deg2rad(alpha), np.deg2rad(beta - beta0)

        ca, sa, sb = np.cos(alpha_r), np.sin(alpha_r), np.sin(beta_r)

        kx = k_tot * ((sd * sx + cd * sb * cx) * ca - (sd * cx - cd * sb * sx) * sa)
        ky = k_tot * ((-cd * sx + sd * sb * cx) * ca + (cd * cx + sd * sb * sx) * sa)

        return kx, ky

    def _inverse_func(kx, ky, kperp=None):
        # cx, sx = np.cos(np.deg2rad(xi)), np.sin(np.deg2rad(xi))  # ξ
        # mask = kx**2 + ky**2 > k_tot_sq + 1e-15
        # kx, ky = np.ma.masked_where(mask, kx), np.ma.masked_where(mask, ky)
        if kperp is None:
            k_sq = k_tot_sq
            k = k_tot
        else:
            k_sq = kx**2 + ky**2 + kperp**2
            k = np.sqrt(k_sq)

        kperp = _kperp_func(k_sq, kx, ky)
        kproj = np.sqrt(np.clip(k_sq - (sd * kx - cd * ky) ** 2, a_min=0, a_max=None))

        alpha = np.arcsin((sx * kproj - cx * (sd * kx - cd * ky)) / k)
        beta = np.arctan((cd * kx + sd * ky) / kperp)

        return np.rad2deg(alpha), np.rad2deg(beta) + beta0

    return _forward_func, _inverse_func


def _kconv_func_type1_da(k_tot, delta=0.0, chi=0.0, chi0=0.0, xi=0.0, xi0=0.0):
    _fwd_2, _inv_2 = _kconv_func_type2_da(k_tot, delta, chi, chi0, xi, xi0)

    def _forward_func(alpha, beta):
        return _fwd_2(-beta, alpha)

    def _inverse_func(kx, ky, kperp=None):
        alpha, beta = _inv_2(kx, ky, kperp)
        return beta, -alpha

    return _forward_func, _inverse_func


def _kconv_func_type2_da(k_tot, delta=0.0, chi=0.0, chi0=0.0, xi=0.0, xi0=0.0):
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))  # δ, azimuth
    cx, sx = np.cos(np.deg2rad(xi - xi0)), np.sin(np.deg2rad(xi - xi0))  # ξ
    cc, sc = np.cos(np.deg2rad(chi - chi0)), np.sin(np.deg2rad(chi - chi0))  # χ

    t11, t12, t13 = cx * cd, cx * sd, -sx
    t21, t22, t23 = sc * sx * cd - cc * sd, sc * sx * sd + cc * cd, sc * cx
    t31, t32, t33 = cc * sx * cd + sc * sd, cc * sx * sd - sc * cd, cc * cx

    k_tot_sq = k_tot**2

    def _forward_func(alpha, beta):
        ar, br = np.deg2rad(alpha), np.deg2rad(beta)

        absq = np.sqrt(ar**2 + br**2)
        scab, cab = _sinc(absq), np.cos(absq)

        # Type I DA
        # kx = k_tot * (
        #     (-ar * cd * cx + br * sd * cc - br * cd * sx * sc) * scab
        #     + (sd * sc + cd * sx * cc) * cab
        # )
        # ky = k_tot * (
        #     (-ar * sd * cx - br * cd * cc - br * sd * sx * sc) * scab
        #     - (cd * sc - sd * sx * cc) * cab
        # )

        # Type II DA
        kx = k_tot * (
            (-br * cd * cx - ar * sd * cc + ar * cd * sx * sc) * scab
            + (sd * sc + cd * sx * cc) * cab
        )
        ky = k_tot * (
            (-br * sd * cx + ar * cd * cc + ar * sd * sx * sc) * scab
            - (cd * sc - sd * sx * cc) * cab
        )

        return kx, ky

    def _inverse_func(kx, ky, kperp=None):
        # mask = kx**2 + ky**2 > k_tot_sq + 1e-15
        # kx, ky = np.ma.masked_where(mask, kx), np.ma.masked_where(mask, ky)

        if kperp is None:
            k_sq = k_tot_sq
            k = k_tot
        else:
            k_sq = kx**2 + ky**2 + kperp**2
            k = np.sqrt(k_sq)

        kperp = _kperp_func(k_sq, kx, ky)  # sqrt(k² - k_x² - k_y²)

        proj1 = t11 * kx + t12 * ky + t13 * kperp
        proj2 = t21 * kx + t22 * ky + t23 * kperp
        proj3 = t31 * kx + t32 * ky + t33 * kperp

        # Type I DA
        # alpha = (
        #     -np.arccos(np.clip(proj3 / k_tot, -1, 1))
        #     * proj1
        #     / np.sqrt((k_sq - proj3**2).clip(min=0))
        # )
        # beta = (
        #     -np.arccos(np.clip(proj3 / k_tot, -1, 1))
        #     * proj2
        #     / np.sqrt((k_sq - proj3**2).clip(min=0))
        # )

        # Type II DA
        alpha = (
            np.arccos(np.clip(proj3 / k, -1, 1))
            * proj2
            / np.sqrt((k_sq - proj3**2).clip(min=0))
        )
        beta = (
            -np.arccos(np.clip(proj3 / k, -1, 1))
            * proj1
            / np.sqrt((k_sq - proj3**2).clip(min=0))
        )

        return np.rad2deg(alpha), np.rad2deg(beta)

    return _forward_func, _inverse_func
