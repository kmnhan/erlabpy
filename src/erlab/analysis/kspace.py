"""Momentum conversion functions.

Typically, the user will not have to call this module directly, but will instead use the
accessor method :meth:`xarray.DataArray.kspace.convert`.

For more front-end utilities related to momentum conversion, see the documentation of
the :attr:`xarray.DataArray.kspace` accessor.

Angle conventions and function forms are based on Ref. :cite:p:`ishida2018kconv`.

"""

from collections.abc import Callable

import numba
import numexpr as ne
import numpy as np
import numpy.typing as npt
import xarray

import erlab.constants
from erlab.constants import AxesConfiguration
from erlab.utils.array import broadcast_args


@numba.vectorize
def _sinc(x):
    return np.sinc(x / np.pi)


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

    :func:`erlab.analysis.kspace.get_kconv_forward`
        Get only the forward function.

    :func:`erlab.analysis.kspace.get_kconv_inverse`
        Get only the inverse function.

    """

    def forward_func(alpha, beta):
        return get_kconv_forward(configuration)(
            alpha, beta, kinetic_energy, **angle_params
        )

    def inverse_func(kx, ky, kperp=None):
        return get_kconv_inverse(configuration)(
            kx, ky, kperp, kinetic_energy, **angle_params
        )

    return forward_func, inverse_func


def get_kconv_forward(configuration: AxesConfiguration | int) -> Callable:
    """Return the appropriate forward momentum conversion function.

    The returned function takes :math:`(α, β, k_{tot})` as mandatory arguments and
    returns :math:`(k_x, k_y)`. Note that :math:`k_{tot}` must be computed from the
    kinetic energy before passing it to the function. The angle parameters can be passed
    as optional arguments with default values of zero.

    Parameters
    ----------
    configuration
        Experimental configuration.

    Returns
    -------
    forward: Callable
        The forward conversion function with the following signature:

        .. code-block:: python

            def forward(alpha, beta, kinetic_energy, **angle_params) -> (kx, ky)
            ...

        ``**angle_params`` are the optional angle parameters. For geometry without DA,
        they are ``delta, xi, xi0, beta0``. For geometry with DA, they are ``delta, chi,
        chi0, xi, xi0``. All angle parameters have default values of zero.

    """
    match configuration:
        case AxesConfiguration.Type1:
            return _kconv_forward_type1
        case AxesConfiguration.Type2:
            return _kconv_forward_type2
        case AxesConfiguration.Type1DA:
            return _kconv_forward_type1_da
        case AxesConfiguration.Type2DA:
            return _kconv_forward_type2_da
        case _:
            raise ValueError(f"Invalid configuration {configuration}")


def get_kconv_inverse(configuration: AxesConfiguration | int) -> Callable:
    r"""Return the appropriate inverse momentum conversion function.

    The returned function takes :math:`(k_x, k_y, k_\perp, E_k)` as mandatory arguments
    and returns :math:`(α, β)`. The angle parameters can be passed as optional arguments
    with default values of zero.

    If :math:`k_\perp` is `None`, it will be computed from the kinetic energy
    :math:`E_k`. Otherwise, the angles will be computed at the given :math:`k_\perp`.

    Parameters
    ----------
    configuration
        Experimental configuration.

    Returns
    -------
    inverse: Callable
        The inverse conversion function with the following signature:

        .. code-block:: python

            def inverse(kx, ky, kperp, kinetic_energy, **angle_params) -> (alpha, beta)
            ...

        ``**angle_params`` are the optional angle parameters. For geometry without DA,
        they are ``delta, xi, xi0, beta0``. For geometry with DA, they are ``delta, chi,
        chi0, xi, xi0``. All angle parameters have default values of zero.

        If ``kperp`` is `None`, it will be computed from ``kinetic_energy``.
    """
    match configuration:
        case AxesConfiguration.Type1:
            return _kconv_inverse_type1
        case AxesConfiguration.Type2:
            return _kconv_inverse_type2
        case AxesConfiguration.Type1DA:
            return _kconv_inverse_type1_da
        case AxesConfiguration.Type2DA:
            return _kconv_inverse_type2_da
        case _:
            raise ValueError(f"Invalid configuration {configuration}")


# To use both numexpr and xarray broadcasting, we use the decorator to broadcast all
# arguments before passing them to the function.

# Note that we compute the trigonometric functions before broadcasting to avoid
# redundant calculations on large broadcasted arrays.


@broadcast_args
def _calc_forward_type1(k_tot, cd, sd, cx, sx, ca, sa, cb, sb):
    kx = ne.evaluate("k_tot * ((sd * sb + cd * sx * cb) * ca - cd * cx * sa)")
    ky = ne.evaluate("k_tot * ((-cd * sb + sd * sx * cb) * ca - sd * cx * sa)")
    return kx, ky


def _kconv_forward_type1(
    alpha, beta, kinetic_energy, delta=0.0, xi=0.0, xi0=0.0, beta0=0.0
):
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))  # δ
    cx, sx = np.cos(np.deg2rad(xi - xi0)), np.sin(np.deg2rad(xi - xi0))  # ξ - ξ0
    alpha_r, beta_r = np.deg2rad(alpha), np.deg2rad(beta - beta0)
    ca, cb = np.cos(alpha_r), np.cos(beta_r)
    sa, sb = np.sin(alpha_r), np.sin(beta_r)
    return _calc_forward_type1(k_tot, cd, sd, cx, sx, ca, sa, cb, sb)


@broadcast_args
def _calc_inverse_type1(kx, ky, k_sq, k, cx, sx, cd, sd, beta0):
    kperp_expr = "sqrt(k_sq - kx**2 - ky**2)"
    alpha = ne.evaluate(f"arcsin((sx * ({kperp_expr}) - cx * (cd * kx + sd * ky)) / k)")
    beta = ne.evaluate(
        "arctan((sd * kx - cd * ky) / "
        f"(sx * (cd * kx + sd * ky) + cx * ({kperp_expr})))"
    )
    return np.rad2deg(alpha), np.rad2deg(beta) + beta0


def _kconv_inverse_type1(
    kx, ky, kperp, kinetic_energy, delta=0.0, xi=0.0, xi0=0.0, beta0=0.0
):
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))  # δ
    cx, sx = np.cos(np.deg2rad(xi - xi0)), np.sin(np.deg2rad(xi - xi0))  # ξ - ξ0
    if kperp is None:
        k_sq = k_tot**2
        k = k_tot
    else:
        k_sq = kx**2 + ky**2 + kperp**2
        k = np.sqrt(k_sq)  # total momentum inside sample

    return _calc_inverse_type1(kx, ky, k_sq, k, cx, sx, cd, sd, beta0)


@broadcast_args
def _calc_forward_type2(k_tot, cd, sd, cx, sx, ca, sa, sb):
    kx = ne.evaluate(
        "k_tot * ((sd * sx + cd * sb * cx) * ca - (sd * cx - cd * sb * sx) * sa)"
    )
    ky = ne.evaluate(
        "k_tot * ((-cd * sx + sd * sb * cx) * ca + (cd * cx + sd * sb * sx) * sa)"
    )
    return kx, ky


def _kconv_forward_type2(
    alpha, beta, kinetic_energy, delta=0.0, xi=0.0, xi0=0.0, beta0=0.0
):
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))  # δ
    cx, sx = np.cos(np.deg2rad(xi - xi0)), np.sin(np.deg2rad(xi - xi0))  # ξ - ξ0
    alpha_r, beta_r = np.deg2rad(alpha), np.deg2rad(beta - beta0)
    ca, sa, sb = np.cos(alpha_r), np.sin(alpha_r), np.sin(beta_r)
    return _calc_forward_type2(k_tot, cd, sd, cx, sx, ca, sa, sb)


@broadcast_args
def _calc_inverse_type2(kx, ky, k_sq, k, cx, sx, cd, sd, beta0):
    kperp_expr = "sqrt(k_sq - kx**2 - ky**2)"
    kproj_expr = "sqrt(k_sq - (sd * kx - cd * ky) ** 2)"
    alpha = ne.evaluate(f"arcsin((sx * ({kproj_expr}) - cx * (sd * kx - cd * ky)) / k)")
    beta = ne.evaluate(f"arctan((cd * kx + sd * ky) / ({kperp_expr}))")
    return np.rad2deg(alpha), np.rad2deg(beta) + beta0


def _kconv_inverse_type2(
    kx, ky, kperp, kinetic_energy, delta=0.0, xi=0.0, xi0=0.0, beta0=0.0
):
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))  # δ
    cx, sx = np.cos(np.deg2rad(xi - xi0)), np.sin(np.deg2rad(xi - xi0))  # ξ - ξ0
    if kperp is None:
        k_sq = k_tot**2
        k = k_tot
    else:
        k_sq = kx**2 + ky**2 + kperp**2
        k = np.sqrt(k_sq)  # total momentum inside sample

    return _calc_inverse_type2(kx, ky, k_sq, k, cx, sx, cd, sd, beta0)


@broadcast_args
def _calc_forward_type2_da(k_tot, cd, sd, cx, sx, cc, sc, ar, br, scab, cab):
    kx = ne.evaluate(
        "k_tot * ((-br * cd * cx - ar * sd * cc + ar * cd * sx * sc) * scab"
        " + (sd * sc + cd * sx * cc) * cab)"
    )
    ky = ne.evaluate(
        "k_tot * ((-br * sd * cx + ar * cd * cc + ar * sd * sx * sc) * scab"
        " - (cd * sc - sd * sx * cc) * cab)"
    )
    return kx, ky


def _kconv_forward_type2_da(
    alpha, beta, kinetic_energy, delta=0.0, chi=0.0, chi0=0.0, xi=0.0, xi0=0.0
):
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))  # δ, azimuth
    cx, sx = np.cos(np.deg2rad(xi - xi0)), np.sin(np.deg2rad(xi - xi0))  # ξ
    cc, sc = np.cos(np.deg2rad(chi - chi0)), np.sin(np.deg2rad(chi - chi0))  # χ
    ar, br = np.deg2rad(alpha), np.deg2rad(beta)
    absq = np.sqrt(ar**2 + br**2)
    scab, cab = _sinc(absq), np.cos(absq)

    return _calc_forward_type2_da(k_tot, cd, sd, cx, sx, cc, sc, ar, br, scab, cab)


@broadcast_args
def _calc_inverse_type2_da(kx, ky, k_sq, k, cd, sd, cx, sx, cc, sc):
    kperp_expr = "sqrt(k_sq - kx**2 - ky**2)"
    t11_expr, t12_expr, t13_expr = "cx * cd", "cx * sd", "-sx"
    t21_expr, t22_expr, t23_expr = (
        "sc * sx * cd - cc * sd",
        "sc * sx * sd + cc * cd",
        "sc * cx",
    )
    t31_expr, t32_expr, t33_expr = (
        "cc * sx * cd + sc * sd",
        "cc * sx * sd - sc * cd",
        "cc * cx",
    )
    proj1_expr = (
        f"({t11_expr}) * kx + ({t12_expr}) * ky + ({t13_expr}) * ({kperp_expr})"
    )
    proj2_expr = (
        f"({t21_expr}) * kx + ({t22_expr}) * ky + ({t23_expr}) * ({kperp_expr})"
    )
    proj3_expr = (
        f"({t31_expr}) * kx + ({t32_expr}) * ky + ({t33_expr}) * ({kperp_expr})"
    )
    alpha = ne.evaluate(
        f"arccos(({proj3_expr}) / k) * ({proj2_expr}) / sqrt(k_sq - ({proj3_expr})**2)"
    )
    beta = ne.evaluate(
        f"-arccos(({proj3_expr}) / k) * ({proj1_expr}) / sqrt(k_sq - ({proj3_expr})**2)"
    )
    return np.rad2deg(alpha), np.rad2deg(beta)


def _kconv_inverse_type2_da(
    kx, ky, kperp, kinetic_energy, delta=0.0, chi=0.0, chi0=0.0, xi=0.0, xi0=0.0
):
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))  # δ, azimuth
    cx, sx = np.cos(np.deg2rad(xi - xi0)), np.sin(np.deg2rad(xi - xi0))  # ξ
    cc, sc = np.cos(np.deg2rad(chi - chi0)), np.sin(np.deg2rad(chi - chi0))  # χ

    if kperp is None:
        k_sq = k_tot**2
        k = k_tot
    else:
        k_sq = kx**2 + ky**2 + kperp**2
        k = np.sqrt(k_sq)

    return _calc_inverse_type2_da(kx, ky, k_sq, k, cd, sd, cx, sx, cc, sc)


def _kconv_forward_type1_da(
    alpha, beta, kinetic_energy, delta=0.0, chi=0.0, chi0=0.0, xi=0.0, xi0=0.0
):
    return _kconv_forward_type2_da(
        -beta, alpha, kinetic_energy, delta, chi, chi0, xi, xi0
    )


def _kconv_inverse_type1_da(
    kx, ky, kperp, kinetic_energy, delta=0.0, chi=0.0, chi0=0.0, xi=0.0, xi0=0.0
):
    alpha, beta = _kconv_inverse_type2_da(
        kx, ky, kperp, kinetic_energy, delta, chi, chi0, xi, xi0
    )
    return beta, -alpha
