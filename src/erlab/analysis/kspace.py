"""Momentum conversion functions.

Typically, the user will not have to call this module directly, but will instead use the
:func:`erlab.accessors.MomentumAccessor.convert` method.

Angle conventions and function forms are based on Ref. :cite:p:`ishida2018kconv`.

"""

import enum
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import xarray

import erlab.constants
import erlab.io
import erlab.lattice


class AxesConfiguration(enum.IntEnum):
    """Enum class representing different types of axes configurations.

    See Ref. :cite:p:`ishida2018kconv`.

    """

    Type1 = 1
    Type2 = 2
    Type1DA = 3
    Type2DA = 4


def _sinc(x):
    return np.sin(x) / (x + 1e-15)


def _kz_func_inv(kz, inner_potential, kx, ky):
    k_perp_sq = kx**2 + ky**2
    k_z_sq = kz**2
    return k_perp_sq + k_z_sq - inner_potential / erlab.constants.rel_kzconv


def _kperp_func(k_tot_sq, kx, ky):
    r""":math:`\sqrt{k^2 - k_x^2 - k_y^2}`."""
    return np.sqrt(np.clip(k_tot_sq - kx**2 - ky**2, a_min=0, a_max=None))


def kz_func(kinetic_energy, inner_potential, kx, ky):
    r"""Calculate the out-of-plane momentum.

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


def get_kconv_func(
    kinetic_energy: float | npt.NDArray | xarray.DataArray,
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
) -> tuple[Callable, Callable]:
    """Return appropriate momentum conversion functions.

    The appropriate function is created by the given configuration and kinetic energy.

    Parameters
    ----------
    kinetic_energy
        The kinetic energy in eV.
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
        Inverse function that takes :math:`(k_x, k_y)` or :math:`(k_x, k_y, k_z)` and
        returns :math:`(α, β)`. If :math:`k_z` is given, it will return the angles
        broadcasted to :math:`k_z` instead of the provided kinetic energy.

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
            ValueError(f"Invalid configuration {configuration}")

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

    def _inverse_func(kx, ky, kz=None):
        # cx, sx = np.cos(np.deg2rad(xi)), np.sin(np.deg2rad(xi))  # ξ
        # mask = kx**2 + ky**2 > k_tot_sq + 1e-15
        # kx, ky = np.ma.masked_where(mask, kx), np.ma.masked_where(mask, ky)
        if kz is None:
            k_sq = k_tot_sq
            k = k_tot
        else:
            k_sq = kx**2 + ky**2 + kz**2
            k = np.sqrt(k_sq)

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

    def _inverse_func(kx, ky, kz=None):
        # cx, sx = np.cos(np.deg2rad(xi)), np.sin(np.deg2rad(xi))  # ξ
        # mask = kx**2 + ky**2 > k_tot_sq + 1e-15
        # kx, ky = np.ma.masked_where(mask, kx), np.ma.masked_where(mask, ky)
        if kz is None:
            k_sq = k_tot_sq
            k = k_tot
        else:
            k_sq = kx**2 + ky**2 + kz**2
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

    def _inverse_func(kx, ky, kz=None):
        alpha, beta = _inv_2(kx, ky, kz)
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

    def _inverse_func(kx, ky, kz=None):
        # mask = kx**2 + ky**2 > k_tot_sq + 1e-15
        # kx, ky = np.ma.masked_where(mask, kx), np.ma.masked_where(mask, ky)

        if kz is None:
            k_sq = k_tot_sq
            k = k_tot
        else:
            k_sq = kx**2 + ky**2 + kz**2
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
