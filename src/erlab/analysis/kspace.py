"""Momentum conversion functions.

Typically, the user will not have to call this module directly, but will instead use the
accessor method :meth:`xarray.DataArray.kspace.convert`.

For more front-end utilities related to momentum conversion, see the documentation of
the :attr:`xarray.DataArray.kspace` accessor.

Angle conventions and function forms are based on Ref. :cite:p:`ishida2018kconv`.

"""

__all__ = [
    "InvalidConfigurationError",
    "change_configuration",
    "exact_cut_alpha",
    "exact_hv_cut_coords",
    "get_kconv_forward",
    "get_kconv_func",
    "get_kconv_inverse",
    "hv_func",
    "kperp_from_kz",
    "kz_func",
]

from collections.abc import Callable

import numba
import numexpr as ne
import numpy as np
import numpy.typing as npt
import xarray

import erlab
import erlab.constants
from erlab.constants import AxesConfiguration
from erlab.utils.array import broadcast_args

_EXACT_CUT_METADATA_HINT = (
    " For normal ARPES data this usually indicates a non-physical angle value or "
    "offset in the metadata; check the angular coordinates and offsets."
)


class InvalidConfigurationError(ValueError):
    """Raised when a momentum-conversion routine receives an invalid configuration."""

    def __init__(
        self,
        configuration,
        *,
        context: str | None = None,
    ) -> None:
        supported_str = ", ".join(cfg.name for cfg in AxesConfiguration)

        try:
            normalized = AxesConfiguration(configuration)
        except ValueError:
            if context is None:
                message = (
                    f"Invalid configuration {configuration!r}. Valid configurations "
                    f"are: {supported_str}."
                )
            else:
                message = (
                    f"{context} received invalid configuration {configuration!r}. "
                    f"Valid configurations are: {supported_str}."
                )
        else:
            if context is None:
                message = f"Unexpected configuration {normalized.name}."
            else:
                message = (
                    f"{context} received unexpected configuration {normalized.name}."
                )

        super().__init__(message)
        self.configuration = configuration
        self.context = context


def _normalize_configuration(
    configuration: AxesConfiguration | int,
    *,
    context: str | None = None,
) -> AxesConfiguration:
    try:
        normalized = AxesConfiguration(configuration)
    except ValueError as exc:
        raise InvalidConfigurationError(configuration, context=context) from exc

    return normalized


def _slit_axis_name(configuration: AxesConfiguration) -> str:
    return (
        "kx"
        if configuration in (AxesConfiguration.Type1, AxesConfiguration.Type1DA)
        else "ky"
    )


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
    new: AxesConfiguration = _normalize_configuration(
        configuration,
        context="change_configuration",
    )

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
    configuration = _normalize_configuration(
        configuration,
        context="get_kconv_forward",
    )
    match configuration:  # pragma: no branch
        case AxesConfiguration.Type1:
            return _kconv_forward_type1
        case AxesConfiguration.Type2:
            return _kconv_forward_type2
        case AxesConfiguration.Type1DA:
            return _kconv_forward_type1_da
        case AxesConfiguration.Type2DA:
            return _kconv_forward_type2_da
        case _:
            raise InvalidConfigurationError(  # pragma: no cover
                configuration, context="get_kconv_forward"
            )


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
    configuration = _normalize_configuration(
        configuration,
        context="get_kconv_inverse",
    )
    match configuration:  # pragma: no branch
        case AxesConfiguration.Type1:
            return _kconv_inverse_type1
        case AxesConfiguration.Type2:
            return _kconv_inverse_type2
        case AxesConfiguration.Type1DA:
            return _kconv_inverse_type1_da
        case AxesConfiguration.Type2DA:
            return _kconv_inverse_type2_da
        case _:
            raise InvalidConfigurationError(  # pragma: no cover
                configuration, context="get_kconv_inverse"
            )


def exact_cut_alpha(
    slit_momentum,
    beta,
    kinetic_energy,
    alpha_reference,
    configuration: AxesConfiguration | int,
    *,
    delta=0.0,
    chi=0.0,
    chi0=0.0,
    xi=0.0,
    xi0=0.0,
    beta0=0.0,
):
    r"""Return the exact sampled :math:`\alpha(k_{\mathrm{slit}}, E_k)` for 2D cuts.

    This function exposes the exact one-dimensional cut inverse used by
    :meth:`xarray.DataArray.kspace.convert` when the dataset has no ``beta`` dimension.

    For slit analyzers (`Type1` / `Type2`), Appendix A of :cite:t:`ishida2018kconv`
    reduces the slit-axis momentum to

    .. math::

        k_{\mathrm{slit}} = k (A \cos \alpha + B \sin \alpha),

    where

    .. math::

        k = \frac{\sqrt{2 m_e E_k}}{\hbar},

    and the coefficients depend only on the geometry and fixed :math:`\bar{\beta} =
    \beta - \beta_0`, :math:`\bar{\xi} = \xi - \xi_0`. The implemented coefficients are

    .. math::

        \text{Type1}: \qquad A = \sin\delta \sin\bar{\beta} + \cos\delta \sin\bar{\xi}
        \cos\bar{\beta}, \qquad B = -\cos\delta \cos\bar{\xi},

    .. math::

        \text{Type2}: \qquad A = -\cos\delta \sin\bar{\xi} + \sin\delta \sin\bar{\beta}
        \cos\bar{\xi}, \qquad B = \cos\delta \cos\bar{\xi} + \sin\delta \sin\bar{\beta}
        \sin\bar{\xi}.

    Writing

    .. math::

        A \cos \alpha + B \sin \alpha = R \cos(\alpha - \psi),

    with :math:`R = \sqrt{A^2 + B^2}` and :math:`\psi = \operatorname{atan}(B, A)`,
    gives the explicit inverse

    .. math::

        \alpha = \psi \pm \cos^{-1}\left(\frac{k_{\mathrm{slit}}}{k R}\right).

    The sign is determined from the measured ``alpha`` interval by replaying the
    normalized source relation on ``alpha_reference``. If the sampled source
    :math:`k_{\mathrm{slit}}(\alpha)` turns around within the measured interval, the
    inverse is multi-valued and the function raises.

    For deflector analyzers (`Type1DA` / `Type2DA`), the exact inverse is written in a
    rotated frame. If the measured momentum vector is rotated back into the deflector
    frame, let :math:`(P_1, P_2, P_3)` denote the resulting momentum components along
    that frame's three orthogonal axes. The implemented DA inverse uses the identities

    .. math::

        P_1 = -k \beta \operatorname{sinc}\eta, \qquad P_2 =  k \alpha
        \operatorname{sinc}\eta, \qquad P_3 =  k \cos\eta,

    with :math:`\eta = \sqrt{\alpha^2 + \beta^2}` where :math:`0 \le \eta < \pi` due to
    the photoemission horizon. For fixed :math:`\beta`, the slit-axis component has the
    form

    .. math::

        k_{\mathrm{slit}} = k \left[A \cos\eta + \left(B \alpha + C
        \beta\right)\operatorname{sinc}\eta\right],

    which does not reduce to an elementary explicit inverse
    :math:`\alpha(k_{\mathrm{slit}})`. The deflector implementation therefore evaluates
    the exact slit-axis curve on the measured ``alpha`` interval and interpolates target
    slit-momentum values onto that monotonic sampled source curve. This is still exact
    in the sense that it does not use small-angle or other geometric approximations; the
    only requirement is that the measured source curve be one-to-one on
    ``alpha_reference``.

    Parameters
    ----------
    slit_momentum
        Target slit-axis momentum values in Å⁻¹. For ``Type1`` and ``Type1DA`` this is
        :math:`k_x`; for ``Type2`` and ``Type2DA`` it is :math:`k_y`.
    beta
        Analyzer angle :math:`\beta` in degrees on the source grid. This may be a
        scalar, an array-like object, or an :class:`xarray.DataArray`. For
        :math:`h\nu`-dependent cuts, ``beta`` may vary along the source ``hv`` axis.
    kinetic_energy
        Kinetic energy :math:`E_k` in eV on the source grid. This may be scalar,
        array-like, or an :class:`xarray.DataArray`.
    alpha_reference
        Measured source ``alpha`` coordinate in degrees. The exact inverse is defined on
        this sampled interval, so branch selection and invertibility are determined from
        this coordinate.
    configuration
        Experimental configuration, one of :class:`erlab.constants.AxesConfiguration`.
    delta
        Sample azimuth :math:`\delta` in degrees.
    chi
        Deflector rotation angle :math:`\chi` in degrees. Used only for ``Type1DA`` and
        ``Type2DA``.
    chi0
        Offset :math:`\chi_0` in degrees. Used only for ``Type1DA`` and ``Type2DA``.
    xi
        Sample tilt :math:`\xi` in degrees.
    xi0
        Offset :math:`\xi_0` in degrees.
    beta0
        Offset :math:`\beta_0` in degrees. Used only for ``Type1`` and ``Type2``.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Exact sampled :math:`\alpha` values on the target slit-momentum grid. The return
        type follows ``slit_momentum``: if it is a DataArray, a DataArray is returned;
        otherwise a NumPy array is returned.

    Raises
    ------
    ValueError
        If the exact source relation is not invertible on the measured interval, if the
        slit-axis momentum becomes effectively constant, or if the target slit momentum
        lies outside the physically reachable domain.

    """
    configuration = _normalize_configuration(
        configuration,
    )
    slit_axis = _slit_axis_name(configuration)

    alpha_reference, alpha_dim = _exact_1d_coord(
        alpha_reference,
        name="alpha_reference",
        default_dim="alpha",
        context="exact cut conversion",
    )

    match configuration:  # pragma: no branch
        case AxesConfiguration.Type1 | AxesConfiguration.Type2:
            slit_momentum, target_is_dataarray = _exact_array(
                slit_momentum,
                name="slit_momentum",
                default_dim=slit_axis,
            )
            A, B = _fixed_beta_slit_coefficients(
                configuration, beta, delta=delta, xi=xi, xi0=xi0, beta0=beta0
            )
            R = np.hypot(A, B)
            degenerate = np.isclose(R, 0.0)
            if isinstance(degenerate, xarray.DataArray):
                has_degenerate_slice = bool(degenerate.any())
            else:
                has_degenerate_slice = bool(np.any(degenerate))
            if has_degenerate_slice:
                raise ValueError(
                    "Exact slit-cut conversion is ill-defined because the slit-axis "
                    "momentum does not vary with alpha for the current geometry."
                    + _EXACT_CUT_METADATA_HINT
                )

            alpha_reference_values = np.asarray(alpha_reference, dtype=float)
            alpha_reference_r = np.deg2rad(alpha_reference)
            psi = np.arctan2(B, A)
            u_reference = A * np.cos(alpha_reference_r) + B * np.sin(alpha_reference_r)
            branch_sign = xarray.apply_ufunc(
                _select_slit_branch_1d,
                alpha_reference_r,
                u_reference / R,
                psi,
                input_core_dims=[[alpha_dim], [alpha_dim], []],
                output_core_dims=[[]],
                dask="parallelized",
                vectorize=True,
                output_dtypes=[np.int8],
            )

            reference_center = float(
                np.deg2rad(
                    np.mean([alpha_reference_values[0], alpha_reference_values[-1]])
                )
            )

            k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
            u = slit_momentum / (k_tot * R)
            tol = 1e-12
            valid = np.isfinite(u) & (np.abs(u) <= 1.0 + tol)
            alpha = _candidate_alpha_from_slit(np.clip(u, -1.0, 1.0), psi, branch_sign)
            alpha = _align_periodic_radians(alpha, reference_center)
            alpha = xarray.where(valid, alpha, np.nan)
            alpha_target = np.rad2deg(alpha)
        case AxesConfiguration.Type1DA | AxesConfiguration.Type2DA:
            slit_momentum, slit_dim, target_is_dataarray = _exact_slit_target_coord(
                slit_momentum,
                slit_axis=slit_axis,
                context="exact cut conversion",
            )
            forward = get_kconv_forward(configuration)
            kx_source, ky_source = forward(
                alpha_reference,
                beta,
                kinetic_energy,
                delta=delta,
                chi=chi,
                chi0=chi0,
                xi=xi,
                xi0=xi0,
            )
            slit_source = (
                kx_source if configuration == AxesConfiguration.Type1DA else ky_source
            )
            alpha_target = _interp_exact_source_curve(
                slit_source,
                alpha_reference,
                slit_momentum,
                source_dim=alpha_dim,
                target_dim=slit_dim,
            )
        case _:
            raise InvalidConfigurationError(  # pragma: no cover
                configuration
            )

    return _unwrap_exact_result(alpha_target, target_is_dataarray=target_is_dataarray)


def exact_hv_cut_coords(
    slit_momentum,
    kz,
    beta,
    hv,
    kinetic_energy,
    alpha_reference,
    configuration: AxesConfiguration | int,
    inner_potential,
    *,
    delta=0.0,
    chi=0.0,
    chi0=0.0,
    xi=0.0,
    xi0=0.0,
    beta0=0.0,
):
    r"""Return exact ``alpha``, ``hv``, and carried momentum targets for 2D cuts.

    This function exposes the exact :math:`h\nu`-dependent cut inversion used by
    :meth:`xarray.DataArray.kspace.convert` when the dataset has no ``beta`` dimension.

    The calculation proceeds in three exact sampled steps.

    1. Recover :math:`\alpha_{\mathrm{source}}(h\nu)` on the measured source grid with
       :func:`exact_cut_alpha`.
    2. Reconstruct the in-plane momentum perpendicular to the slit on that same source
       grid. For slit analyzers this is

       .. math::

           k_{\mathrm{other}} = k(C \cos\alpha + D \sin\alpha),

       with geometry-dependent coefficients :math:`C` and :math:`D`. For deflector
       analyzers, :math:`k_x` and :math:`k_y` are reconstructed directly from the exact
       forward transform.
    3. Build the exact sampled out-of-plane source curve from the free-electron
       final-state relation

    .. math::

        k_z(h\nu)^2 = \frac{2 m_e}{\hbar^2}\left(E_k(h\nu) + V_0\right) - k_x(h\nu)^2 -
        k_y(h\nu)^2,

    with

    .. math::

        E_k(h\nu) = h\nu - \Phi + E_b,

    where :math:`\Phi` is the work function and :math:`E_b` is the binding energy.

    The implementation then inverts the sampled monotonic curve :math:`k_z(h\nu)` along
    the measured ``hv`` axis to obtain ``hv_target`` and uses the same interpolation to
    obtain ``alpha_target`` and the carried orthogonal in-plane momentum
    ``other_target`` on the target grid. As with :func:`exact_cut_alpha`, invertibility
    still requires the sampled source relation to remain one-to-one over the measured
    interval.

    Parameters
    ----------
    slit_momentum
        Target slit-axis momentum values in Å⁻¹. For ``Type1`` and ``Type1DA`` this is
        :math:`k_x`; for ``Type2`` and ``Type2DA`` it is :math:`k_y`.
    kz
        Target :math:`k_z` values in Å⁻¹.
    beta
        Analyzer angle :math:`\beta` in degrees on the source grid. This may be a
        scalar, an array-like object, or an :class:`xarray.DataArray`.
    hv
        Measured photon-energy coordinate in eV. This defines the sampled source axis
        along which the exact :math:`k_z(h\nu)` relation is inverted.
    kinetic_energy
        Kinetic energy :math:`E_k` in eV on the source grid.
    alpha_reference
        Measured source ``alpha`` coordinate in degrees.
    configuration
        Experimental configuration, one of :class:`erlab.constants.AxesConfiguration`.
    inner_potential
        Inner potential :math:`V_0` in eV.
    delta
        Sample azimuth :math:`\delta` in degrees.
    chi
        Deflector rotation angle :math:`\chi` in degrees. Used only for ``Type1DA`` and
        ``Type2DA``.
    chi0
        Offset :math:`\chi_0` in degrees. Used only for ``Type1DA`` and ``Type2DA``.
    xi
        Sample tilt :math:`\xi` in degrees.
    xi0
        Offset :math:`\xi_0` in degrees.
    beta0
        Offset :math:`\beta_0` in degrees. Used only for ``Type1`` and ``Type2``.

    Returns
    -------
    tuple of xarray.DataArray
        ``(alpha_target, hv_target, other_target)`` where ``other_target`` is the
        carried in-plane momentum perpendicular to the slit.

    Raises
    ------
    ValueError
        If the exact sampled source relation is not invertible on the measured interval.

    """
    configuration = _normalize_configuration(configuration)

    slit_axis = _slit_axis_name(configuration)
    slit_momentum, _, _ = _exact_slit_target_coord(
        slit_momentum,
        slit_axis=slit_axis,
        context="exact hv-cut conversion",
    )
    hv, hv_dim = _exact_1d_coord(
        hv,
        name="hv",
        default_dim="hv",
        context="exact hv-cut conversion",
    )
    kz, kz_dim = _exact_1d_coord(
        kz,
        name="kz",
        default_dim="kz",
        context="exact hv-cut conversion",
    )

    alpha_source = exact_cut_alpha(
        slit_momentum,
        beta,
        kinetic_energy,
        alpha_reference,
        configuration,
        delta=delta,
        chi=chi,
        chi0=chi0,
        xi=xi,
        xi0=xi0,
        beta0=beta0,
    )
    other_source = _exact_other_axis_momentum(
        alpha_source,
        beta,
        kinetic_energy,
        configuration,
        delta=delta,
        chi=chi,
        chi0=chi0,
        xi=xi,
        xi0=xi0,
        beta0=beta0,
    )

    if slit_axis == "kx":
        kx_source = slit_momentum
        ky_source = other_source
    else:
        kx_source = other_source
        ky_source = slit_momentum

    kz_source = kz_func(kinetic_energy, inner_potential, kx_source, ky_source)
    hv_target = _interp_exact_source_curve(
        kz_source,
        hv,
        kz,
        source_dim=hv_dim,
        target_dim=kz_dim,
    )
    alpha_target = _interp_exact_source_curve(
        kz_source,
        alpha_source,
        kz,
        source_dim=hv_dim,
        target_dim=kz_dim,
    )

    alpha_target = _transpose_exact_target(
        alpha_target,
        slit_axis,
        kz_dim,
        kinetic_energy,
        excluded_dim=hv_dim,
    )
    hv_target = _transpose_exact_target(
        hv_target,
        slit_axis,
        kz_dim,
        kinetic_energy,
        excluded_dim=hv_dim,
    )
    carried_source = ky_source if slit_axis == "kx" else kx_source
    other_target = _transpose_exact_target(
        _interp_exact_source_curve(
            kz_source,
            carried_source,
            kz,
            source_dim=hv_dim,
            target_dim=kz_dim,
        ),
        slit_axis,
        kz_dim,
        kinetic_energy,
        excluded_dim=hv_dim,
    )

    return alpha_target, hv_target, other_target


def _offsets_from_normal_emission(
    configuration: AxesConfiguration | int,
    alpha: float,
    beta: float,
    *,
    xi: float,
    chi: float | None = None,
) -> dict[str, float]:
    configuration = _normalize_configuration(configuration)

    match configuration:  # pragma: no branch
        case AxesConfiguration.Type1 | AxesConfiguration.Type2:
            return {"xi": xi - alpha, "beta": beta}
        case AxesConfiguration.Type1DA | AxesConfiguration.Type2DA:
            if chi is None:
                raise ValueError("`chi` is required for deflector configurations.")

            if configuration == AxesConfiguration.Type1DA:
                alpha, beta = -beta, alpha

            alpha_rad = np.deg2rad(alpha)
            beta_rad = np.deg2rad(beta)
            radius = np.hypot(alpha_rad, beta_rad)
            sinc_radius = _sinc(radius)

            x = -beta_rad * sinc_radius
            y = alpha_rad * sinc_radius
            z = np.cos(radius)

            xi_delta = np.rad2deg(np.arcsin(np.clip(-x, -1.0, 1.0)))
            chi_delta = np.rad2deg(np.arctan2(y, z))

            return {"xi": xi - xi_delta, "chi": chi - chi_delta}
        case _:
            raise InvalidConfigurationError(  # pragma: no cover
                configuration
            )


def _normal_emission_from_angle_params(
    configuration: AxesConfiguration | int,
    angle_params: dict[str, float],
) -> tuple[float, float]:
    configuration = _normalize_configuration(configuration)

    match configuration:  # pragma: no branch
        case AxesConfiguration.Type1 | AxesConfiguration.Type2:
            return angle_params["xi"] - angle_params["xi0"], angle_params["beta0"]
        case AxesConfiguration.Type1DA | AxesConfiguration.Type2DA:
            xi_delta = np.deg2rad(angle_params["xi"] - angle_params["xi0"])
            chi_delta = np.deg2rad(angle_params["chi"] - angle_params["chi0"])

            x = -np.sin(xi_delta)
            yz_norm = np.cos(xi_delta)
            y = yz_norm * np.sin(chi_delta)
            z = yz_norm * np.cos(chi_delta)

            radius = np.arccos(np.clip(z, -1.0, 1.0))
            if abs(radius) < 1e-12:
                alpha = beta = 0.0
            else:
                sinc_radius = _sinc(radius)
                alpha = np.rad2deg(y / sinc_radius)
                beta = np.rad2deg(-x / sinc_radius)

            if configuration == AxesConfiguration.Type1DA:
                return beta, -alpha
            return alpha, beta
        case _:
            raise InvalidConfigurationError(  # pragma: no cover
                configuration
            )


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
        f"arctan2(sd * kx - cd * ky, sx * (cd * kx + sd * ky) + cx * ({kperp_expr}))"
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
    beta = ne.evaluate(f"arctan2(cd * kx + sd * ky, {kperp_expr})")
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


def _fixed_beta_slit_coefficients(
    configuration: AxesConfiguration | int,
    beta,
    *,
    delta=0.0,
    xi=0.0,
    xi0=0.0,
    beta0=0.0,
):
    r"""Return the slit-axis coefficients for fixed-:math:`\beta` cuts.

    Parameters
    ----------
    configuration
        Experimental configuration. Only type I and type II are supported.
    beta
        Fixed analyzer angle :math:`\beta` in degrees.
    delta
        Sample azimuth :math:`\delta` in degrees.
    xi
        Sample tilt :math:`\xi` in degrees.
    xi0
        Offset :math:`\xi_0` in degrees.
    beta0
        Offset :math:`\beta_0` in degrees.

    Returns
    -------
    tuple
        The coefficients :math:`A` and :math:`B` such that the slit-axis momentum is
        :math:`k_{\mathrm{slit}} = k (A \cos \alpha + B \sin \alpha)`. If any input is
        an :class:`xarray.DataArray`, the returned coefficients preserve its labeled
        broadcast dimensions.

    """
    configuration = AxesConfiguration(configuration)

    delta_r = np.deg2rad(delta)
    beta_bar = np.deg2rad(beta - beta0)
    xi_bar = np.deg2rad(xi - xi0)

    cd, sd = np.cos(delta_r), np.sin(delta_r)
    cb, sb = np.cos(beta_bar), np.sin(beta_bar)
    cx, sx = np.cos(xi_bar), np.sin(xi_bar)

    match configuration:  # pragma: no branch
        case AxesConfiguration.Type1:
            return sd * sb + cd * sx * cb, -cd * cx
        case AxesConfiguration.Type2:
            return -cd * sx + sd * sb * cx, cd * cx + sd * sb * sx
        case _:
            raise InvalidConfigurationError(  # pragma: no cover
                configuration
            )


def _fixed_beta_other_axis_coefficients(
    configuration: AxesConfiguration | int,
    beta,
    *,
    delta=0.0,
    xi=0.0,
    xi0=0.0,
    beta0=0.0,
):
    r"""Return coefficients for the in-plane momentum perpendicular to the slit.

    The returned coefficients :math:`C` and :math:`D` satisfy

    .. math::

        k_{\mathrm{other}} = k (C \cos\alpha + D \sin\alpha)

    for fixed :math:`\beta`.
    """
    configuration = AxesConfiguration(configuration)

    delta_r = np.deg2rad(delta)
    beta_bar = np.deg2rad(beta - beta0)
    xi_bar = np.deg2rad(xi - xi0)

    cd, sd = np.cos(delta_r), np.sin(delta_r)
    cb, sb = np.cos(beta_bar), np.sin(beta_bar)
    cx, sx = np.cos(xi_bar), np.sin(xi_bar)

    match configuration:  # pragma: no branch
        case AxesConfiguration.Type1:
            return -cd * sb + sd * sx * cb, -sd * cx
        case AxesConfiguration.Type2:
            return sd * sx + cd * sb * cx, -sd * cx + cd * sb * sx
        case _:
            raise InvalidConfigurationError(  # pragma: no cover
                configuration
            )


def _align_periodic_radians(angle, reference):
    return angle + 2 * np.pi * np.round((reference - angle) / (2 * np.pi))


def _candidate_alpha_from_slit(u, psi: float, sign: int, reference=None):
    alpha = psi + sign * np.arccos(np.clip(u, -1.0, 1.0))
    if reference is not None:
        return _align_periodic_radians(alpha, reference)
    return alpha


def _select_slit_branch_1d(alpha_reference_r, u_reference_norm, psi) -> np.int8:
    alpha_reference_r = np.asarray(alpha_reference_r, dtype=float)
    u_reference_norm = np.asarray(u_reference_norm, dtype=float)
    psi = float(np.asarray(psi, dtype=float))

    if alpha_reference_r.ndim != 1 or u_reference_norm.ndim != 1:
        raise ValueError(
            "_select_slit_branch_1d expects one-dimensional reference slices."
        )
    if alpha_reference_r.shape != u_reference_norm.shape:
        raise ValueError(
            "Reference alpha and normalized slit momentum must have matching shapes."
        )
    if alpha_reference_r.size < 2:
        raise ValueError(
            "Exact slit-cut conversion requires at least two alpha reference points."
        )

    du = np.diff(u_reference_norm, axis=-1)
    tol = 1e-12
    if not (np.all(du >= -tol) or np.all(du <= tol)):
        raise ValueError(
            "Exact slit-cut conversion is not invertible over the measured alpha "
            "range because the slit-axis momentum turns around within that interval."
            + _EXACT_CUT_METADATA_HINT
        )
    if np.all(np.abs(du) <= tol):
        raise ValueError(
            "Exact slit-cut conversion is ill-defined because the measured alpha range "
            "maps to an effectively constant slit momentum." + _EXACT_CUT_METADATA_HINT
        )

    ref_plus = _candidate_alpha_from_slit(u_reference_norm, psi, 1, alpha_reference_r)
    ref_minus = _candidate_alpha_from_slit(u_reference_norm, psi, -1, alpha_reference_r)

    err_plus = np.nanmax(np.abs(ref_plus - alpha_reference_r))
    err_minus = np.nanmax(np.abs(ref_minus - alpha_reference_r))
    if not np.isfinite(err_plus) or not np.isfinite(err_minus):
        raise ValueError(
            "Exact slit-cut conversion could not determine a valid inverse branch."
            + _EXACT_CUT_METADATA_HINT
        )

    return np.int8(1 if err_plus <= err_minus else -1)


def _select_slit_branch(alpha_reference_r, u_reference_norm, psi) -> np.ndarray:
    return np.vectorize(_select_slit_branch_1d, signature="(n),(n),()->()")(
        alpha_reference_r,
        u_reference_norm,
        psi,
    ).astype(np.int8)


def _interp_monotonic_interpn_1d(
    source_coord, source_values, target_coord
) -> np.ndarray:
    source_coord = np.asarray(source_coord, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    target_coord = np.asarray(target_coord, dtype=float)

    out = np.full(target_coord.shape, np.nan, dtype=float)

    finite = np.isfinite(source_coord) & np.isfinite(source_values)
    source_coord = source_coord[finite]
    source_values = source_values[finite]

    if source_coord.size < 2:
        return out

    dif = np.diff(source_coord)
    tol = 1e-12
    if not erlab.utils.array.is_monotonic(source_coord, strict=True):
        if np.all(np.abs(dif) <= tol):
            raise ValueError(
                "Exact cut conversion is ill-defined because the exact sampled "
                "source curve is effectively constant over the measured interval."
                + _EXACT_CUT_METADATA_HINT
            )
        if not (np.all(dif >= -tol) or np.all(dif <= tol)):
            raise ValueError(
                "Exact cut conversion is not invertible because the exact sampled "
                "source curve turns around over the measured interval."
                + _EXACT_CUT_METADATA_HINT
            )
        raise ValueError(
            "Exact cut conversion is ill-defined because the exact sampled source "
            "curve is not strictly monotonic over the measured interval."
            + _EXACT_CUT_METADATA_HINT
        )

    if source_coord[0] > source_coord[-1]:
        source_coord = source_coord[::-1]
        source_values = source_values[::-1]

    return np.interp(
        target_coord,
        source_coord,
        source_values,
        left=np.nan,
        right=np.nan,
    )


def _interp_exact_source_curve(
    source_coord: xarray.DataArray,
    source_values,
    target_coord: xarray.DataArray,
    *,
    source_dim: str,
    target_dim: str | None,
) -> xarray.DataArray:
    input_core_dims: list[list[str]] = [[source_dim], [source_dim], []]
    output_core_dims: list[list[str]] = [[]]
    dask_gufunc_kwargs: dict[str, dict[str, int]] | None = None
    if target_dim is not None:
        input_core_dims[2] = [target_dim]
        output_core_dims[0] = [target_dim]
        dask_gufunc_kwargs = {
            "output_sizes": {target_dim: target_coord.sizes[target_dim]}
        }

    return xarray.apply_ufunc(
        _interp_monotonic_interpn_1d,
        source_coord,
        source_values,
        target_coord,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        output_dtypes=[np.float64],
    )


def _exact_1d_coord(
    values,
    *,
    name: str,
    default_dim: str,
    context: str,
) -> tuple[xarray.DataArray, str]:
    if isinstance(values, xarray.DataArray):
        if values.ndim != 1:
            raise ValueError(f"`{name}` must be one-dimensional for {context}.")
        return values, str(values.dims[0])

    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional for {context}.")
    return xarray.DataArray(values, dims=(default_dim,)), default_dim


def _exact_array(
    values,
    *,
    name: str,
    default_dim: str,
) -> tuple[xarray.DataArray, bool]:
    if isinstance(values, xarray.DataArray):
        return values, True

    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return xarray.DataArray(values), False
    if values.ndim == 1:
        return xarray.DataArray(values, dims=(default_dim,)), False
    return (
        xarray.DataArray(
            values,
            dims=tuple(f"_{name}_dim_{i}" for i in range(values.ndim)),
        ),
        False,
    )


def _exact_slit_target_coord(
    slit_momentum,
    *,
    slit_axis: str,
    context: str,
) -> tuple[xarray.DataArray, str | None, bool]:
    if isinstance(slit_momentum, xarray.DataArray):
        if slit_axis in slit_momentum.dims:
            return slit_momentum, slit_axis, True
        if slit_momentum.ndim == 1:
            return slit_momentum, str(slit_momentum.dims[0]), True
        raise ValueError(
            "`slit_momentum` must be one-dimensional or contain the slit-axis "
            f"target dimension for {context}."
        )

    slit_momentum, _ = _exact_array(
        slit_momentum,
        name=f"{slit_axis}_target",
        default_dim=slit_axis,
    )
    return slit_momentum, (slit_axis if slit_momentum.ndim == 1 else None), False


def _unwrap_exact_result(result, *, target_is_dataarray: bool):
    if target_is_dataarray:
        return result
    if isinstance(result, xarray.DataArray):
        return result.values
    return result


def _transpose_exact_target(
    target: xarray.DataArray,
    primary_dim: str,
    secondary_dim: str | None,
    kinetic_energy,
    *,
    excluded_dim: str,
) -> xarray.DataArray:
    preferred_dims: list[str] = []
    if primary_dim in target.dims:
        preferred_dims.append(primary_dim)
    if secondary_dim is not None and secondary_dim in target.dims:
        preferred_dims.append(secondary_dim)
    if isinstance(kinetic_energy, xarray.DataArray):
        preferred_dims.extend(
            [
                str(d)
                for d in kinetic_energy.dims
                if d != excluded_dim and d in target.dims and d not in preferred_dims
            ]
        )
    preferred_dims.extend([str(d) for d in target.dims if d not in preferred_dims])
    return target.transpose(*preferred_dims)


def _fixed_beta_other_axis_momentum(
    alpha,
    beta,
    kinetic_energy,
    configuration: AxesConfiguration | int,
    *,
    delta=0.0,
    xi=0.0,
    xi0=0.0,
    beta0=0.0,
):
    r"""Return the in-plane momentum perpendicular to the slit for fixed-:math:`\beta`.

    This evaluates

    .. math::

        k_{\mathrm{other}} = k (C \cos\alpha + D \sin\alpha)

    using the exact type-I/type-II coefficients for fixed-:math:`\beta` slit cuts.
    Inputs may be NumPy arrays or :class:`xarray.DataArray` objects; labeled
    broadcasting is preserved when available.
    """
    C, D = _fixed_beta_other_axis_coefficients(
        configuration, beta, delta=delta, xi=xi, xi0=xi0, beta0=beta0
    )
    k_tot = erlab.constants.rel_kconv * np.sqrt(kinetic_energy)
    alpha_r = np.deg2rad(alpha)
    return k_tot * (C * np.cos(alpha_r) + D * np.sin(alpha_r))


def _exact_other_axis_momentum(
    alpha,
    beta,
    kinetic_energy,
    configuration: AxesConfiguration | int,
    *,
    delta=0.0,
    chi=0.0,
    chi0=0.0,
    xi=0.0,
    xi0=0.0,
    beta0=0.0,
):
    """Return the carried in-plane momentum perpendicular to the slit for exact cuts."""
    configuration = _normalize_configuration(configuration)
    match configuration:  # pragma: no branch
        case AxesConfiguration.Type1 | AxesConfiguration.Type2:
            return _fixed_beta_other_axis_momentum(
                alpha,
                beta,
                kinetic_energy,
                configuration,
                delta=delta,
                xi=xi,
                xi0=xi0,
                beta0=beta0,
            )
        case AxesConfiguration.Type1DA | AxesConfiguration.Type2DA:
            kx, ky = get_kconv_forward(configuration)(
                alpha,
                beta,
                kinetic_energy,
                delta=delta,
                chi=chi,
                chi0=chi0,
                xi=xi,
                xi0=xi0,
            )
            return ky if configuration == AxesConfiguration.Type1DA else kx
        case _:
            raise InvalidConfigurationError(  # pragma: no cover
                configuration
            )
