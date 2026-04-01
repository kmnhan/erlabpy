import warnings
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

import erlab.analysis
import erlab.constants
from erlab.constants import AxesConfiguration

kinetic = np.array([1.0, 2.0, 3.0])


def _manual_monotonic_interp(
    source_coord: np.ndarray, source_values: np.ndarray, target_coord: np.ndarray
) -> np.ndarray:
    finite = np.isfinite(source_coord) & np.isfinite(source_values)
    source_coord = np.asarray(source_coord[finite], dtype=float)
    source_values = np.asarray(source_values[finite], dtype=float)
    target_coord = np.asarray(target_coord, dtype=float)

    out = np.full(target_coord.shape, np.nan, dtype=float)
    if source_coord.size < 2:
        return out

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


def _generate_funclist() -> list[tuple[Callable, Callable]]:
    funcs = []

    for delta, xi, xi0, beta0 in zip(
        [0, 90.0, -90.0],
        [0, 30.0, -30.0],
        [0.0, 10.0, -10.0],
        [0.0, 10.0, -10.0],
        strict=True,
    ):
        for config in (1, 2):
            funcs.append(  # noqa: PERF401
                erlab.analysis.kspace.get_kconv_func(
                    kinetic,
                    config,
                    {"delta": delta, "xi": xi, "xi0": xi0, "beta0": beta0},
                )
            )
    for delta, chi, chi0, xi, xi0 in zip(
        [0, 90.0, -90.0],
        [0, 10.0, -30.0],
        [0.0, 10.0, -10.0],
        [0.0, 10.0, -10.0],
        [0.0, 10.0, -10.0],
        strict=True,
    ):
        for config in (3, 4):
            funcs.append(  # noqa: PERF401
                erlab.analysis.kspace.get_kconv_func(
                    kinetic,
                    config,
                    {"delta": delta, "chi": chi, "chi0": chi0, "xi": xi, "xi0": xi0},
                )
            )

    return funcs


def test_transform() -> None:
    for forward_func, inverse_func in _generate_funclist():
        alpha, beta = inverse_func(*forward_func(10.0, 20.0))
        assert alpha.size == beta.size == kinetic.size
        assert np.allclose(alpha, 10.0)
        assert np.allclose(beta, 20.0)

        kx, ky = forward_func(*inverse_func(0.1, 0.2))
        assert kx.size == ky.size == kinetic.size
        assert np.allclose(kx, 0.1)
        assert np.allclose(ky, 0.2)


def test_invalid_configuration_error_for_invalid_value() -> None:
    with pytest.raises(
        erlab.analysis.kspace.InvalidConfigurationError,
        match="get_kconv_forward received invalid configuration 999",
    ):
        erlab.analysis.kspace.get_kconv_forward(999)


@pytest.mark.parametrize(
    ("configuration", "context", "expected"),
    [
        pytest.param(
            999, None, "Invalid configuration 999.", id="invalid-without-context"
        ),
        pytest.param(
            AxesConfiguration.Type1,
            None,
            "Unexpected configuration Type1.",
            id="unexpected-without-context",
        ),
        pytest.param(
            AxesConfiguration.Type1,
            "exact cut conversion",
            "exact cut conversion received unexpected configuration Type1.",
            id="unexpected-with-context",
        ),
    ],
)
def test_invalid_configuration_error_message_variants(
    configuration, context: str | None, expected: str
) -> None:
    err = erlab.analysis.kspace.InvalidConfigurationError(
        configuration, context=context
    )

    assert str(err).startswith(expected)


def test_inverse_type1_beta_uses_quadrant_from_arctan2() -> None:
    kx, ky = 3.0, -1.0
    k_sq, k = 25.0, 5.0
    cx, sx = -1.0, 0.0
    cd, sd = 1.0, 0.0
    beta0 = 0.0

    _, beta = erlab.analysis.kspace._calc_inverse_type1(
        kx, ky, k_sq, k, cx, sx, cd, sd, beta0
    )

    kperp = np.sqrt(k_sq - kx**2 - ky**2)
    num = sd * kx - cd * ky
    den = sx * (cd * kx + sd * ky) + cx * kperp
    expected = np.rad2deg(np.arctan2(num, den)) + beta0

    assert np.isclose(beta, expected)
    # This case previously landed in the wrong branch near -14.5°.
    assert beta > 90.0


@pytest.mark.parametrize(
    ("configuration", "angle_params", "beta"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            {"delta": 12.5, "xi": 7.25, "xi0": 2.5, "beta0": -3.75},
            4.5,
            id="type1",
        ),
        pytest.param(
            AxesConfiguration.Type2,
            {"delta": -8.0, "xi": -6.5, "xi0": -1.75, "beta0": 2.25},
            -3.0,
            id="type2",
        ),
    ],
)
def test_fixed_beta_slit_coefficients_reduce_forward_equation(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
    beta: float,
) -> None:
    alpha = np.linspace(-15.0, 15.0, 51)
    kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha, beta, 1.0, **angle_params
    )
    slit_axis = "kx" if configuration == AxesConfiguration.Type1 else "ky"
    slit_values = kx if slit_axis == "kx" else ky

    A, B = erlab.analysis.kspace._fixed_beta_slit_coefficients(
        configuration, beta, **angle_params
    )
    k_tot = erlab.constants.rel_kconv
    expected = k_tot * (A * np.cos(np.deg2rad(alpha)) + B * np.sin(np.deg2rad(alpha)))

    assert np.allclose(slit_values, expected)


@pytest.mark.parametrize(
    ("configuration", "angle_params", "beta"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            {"delta": 12.5, "xi": 7.25, "xi0": 2.5, "beta0": -3.75},
            4.5,
            id="type1",
        ),
        pytest.param(
            AxesConfiguration.Type2,
            {"delta": -8.0, "xi": -6.5, "xi0": -1.75, "beta0": 2.25},
            -3.0,
            id="type2",
        ),
    ],
)
def test_fixed_beta_other_axis_coefficients_reduce_forward_equation(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
    beta: float,
) -> None:
    alpha = np.linspace(-15.0, 15.0, 51)
    kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha, beta, 1.0, **angle_params
    )
    other_values = ky if configuration == AxesConfiguration.Type1 else kx

    C, D = erlab.analysis.kspace._fixed_beta_other_axis_coefficients(
        configuration, beta, **angle_params
    )
    k_tot = erlab.constants.rel_kconv
    expected = k_tot * (C * np.cos(np.deg2rad(alpha)) + D * np.sin(np.deg2rad(alpha)))

    assert np.allclose(other_values, expected)


@pytest.mark.parametrize("descending", [False, True], ids=["ascending", "descending"])
@pytest.mark.parametrize(
    ("configuration", "angle_params", "beta"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            {"delta": 12.5, "xi": 7.25, "xi0": 2.5, "beta0": -3.75},
            4.5,
            id="type1",
        ),
        pytest.param(
            AxesConfiguration.Type2,
            {"delta": -8.0, "xi": -6.5, "xi0": -1.75, "beta0": 2.25},
            -3.0,
            id="type2",
        ),
    ],
)
def test_exact_slit_cut_alpha_roundtrips_reference_cut(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
    beta: float,
    descending: bool,
) -> None:
    alpha = np.linspace(-15.0, 15.0, 61)
    if descending:
        alpha = alpha[::-1]

    kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha, beta, 1.0, **angle_params
    )
    slit_values = kx if configuration == AxesConfiguration.Type1 else ky

    recovered = erlab.analysis.kspace.exact_cut_alpha(
        slit_values,
        beta,
        1.0,
        alpha,
        configuration,
        **angle_params,
    )

    assert np.allclose(recovered, alpha)


def test_exact_slit_cut_alpha_returns_nan_out_of_domain() -> None:
    recovered = erlab.analysis.kspace.exact_cut_alpha(
        np.array([0.0, 10.0]),
        4.5,
        1.0,
        np.linspace(-10.0, 10.0, 21),
        AxesConfiguration.Type1,
        delta=12.5,
        xi=7.25,
        xi0=2.5,
        beta0=-3.75,
    )

    assert np.isfinite(recovered[0])
    assert np.isnan(recovered[1])


def test_exact_slit_cut_alpha_rejects_multivalued_reference_cut() -> None:
    alpha = np.linspace(-15.0, 15.0, 61)
    kx, _ = erlab.analysis.kspace.get_kconv_forward(AxesConfiguration.Type1)(
        alpha,
        0.0,
        1.0,
        delta=0.0,
        xi=90.0,
        xi0=0.0,
        beta0=0.0,
    )

    with pytest.raises(ValueError, match="non-physical angle value or offset"):
        erlab.analysis.kspace.exact_cut_alpha(
            kx,
            0.0,
            1.0,
            alpha,
            AxesConfiguration.Type1,
            delta=0.0,
            xi=90.0,
            xi0=0.0,
            beta0=0.0,
        )


@pytest.mark.parametrize(
    ("configuration", "angle_params", "descending"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            {"delta": 12.5, "xi": 7.25, "xi0": 2.5, "beta0": -3.75},
            False,
            id="type1-ascending",
        ),
        pytest.param(
            AxesConfiguration.Type2,
            {"delta": -80.0, "xi": -80.0, "xi0": 0.0, "beta0": 0.0},
            True,
            id="type2-descending",
        ),
    ],
)
def test_exact_slit_cut_alpha_broadcasts_over_beta_and_kinetic_energy(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
    descending: bool,
) -> None:
    alpha_reference = xr.DataArray(np.linspace(-15.0, 15.0, 61), dims=("alpha",))
    if descending:
        alpha_reference = alpha_reference[::-1]

    beta = xr.DataArray(
        np.array([-20.0, -5.0, 10.0]),
        dims=("hv",),
        coords={"hv": [20, 30, 40]},
    )
    kinetic_energy = xr.DataArray(
        np.array([[22.0, 22.5, 23.0], [24.0, 24.5, 25.0], [26.0, 26.5, 27.0]]),
        dims=("hv", "eV"),
        coords={"hv": beta.hv, "eV": [-0.1, 0.0, 0.1]},
    )

    kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha_reference,
        beta,
        kinetic_energy,
        **angle_params,
    )
    slit_values = kx if configuration == AxesConfiguration.Type1 else ky

    recovered = erlab.analysis.kspace.exact_cut_alpha(
        slit_values,
        beta,
        kinetic_energy,
        alpha_reference,
        configuration,
        **angle_params,
    )
    expected = xr.broadcast(kinetic_energy, alpha_reference)[1]

    xr.testing.assert_allclose(
        recovered.transpose("hv", "eV", "alpha"),
        expected.transpose("hv", "eV", "alpha"),
    )


def test_exact_slit_cut_alpha_rejects_multivalued_broadcast_slice() -> None:
    alpha_reference = xr.DataArray(np.linspace(-15.0, 15.0, 61), dims=("alpha",))
    beta = xr.DataArray(np.array([-5.0, 0.0]), dims=("hv",), coords={"hv": [20, 30]})
    kinetic_energy = xr.DataArray(
        np.array([[22.0, 22.5], [24.0, 24.5]]),
        dims=("hv", "eV"),
        coords={"hv": beta.hv, "eV": [-0.05, 0.05]},
    )

    kx, _ = erlab.analysis.kspace.get_kconv_forward(AxesConfiguration.Type1)(
        alpha_reference,
        beta,
        kinetic_energy,
        delta=-85.0,
        xi=-85.0,
        xi0=0.0,
        beta0=0.0,
    )

    with pytest.raises(ValueError, match="non-physical angle value or offset"):
        erlab.analysis.kspace.exact_cut_alpha(
            kx,
            beta,
            kinetic_energy,
            alpha_reference,
            AxesConfiguration.Type1,
            delta=-85.0,
            xi=-85.0,
            xi0=0.0,
            beta0=0.0,
        )


@pytest.mark.parametrize(
    ("configuration", "angle_params"),
    [
        pytest.param(
            AxesConfiguration.Type1DA,
            {"delta": 9.0, "chi": 6.5, "chi0": 2.0, "xi": 5.25, "xi0": 1.75},
            id="type1da",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            {"delta": -5.0, "chi": -4.0, "chi0": 1.25, "xi": 6.75, "xi0": 0.75},
            id="type2da",
        ),
    ],
)
def test_da_forward_inverse_roundtrips_principal_branch(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
) -> None:
    alpha = np.array([-12.0, -5.0, 0.0, 6.0, 11.0])
    beta = np.array([-6.0, -2.0, 0.0, 3.0, 7.0])
    kinetic_energy = np.linspace(21.0, 25.0, alpha.size)

    kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha, beta, kinetic_energy, **angle_params
    )
    recovered_alpha, recovered_beta = erlab.analysis.kspace.get_kconv_inverse(
        configuration
    )(kx, ky, None, kinetic_energy, **angle_params)

    assert np.allclose(recovered_alpha, alpha)
    assert np.allclose(recovered_beta, beta)


@pytest.mark.parametrize("descending", [False, True], ids=["ascending", "descending"])
@pytest.mark.parametrize(
    ("configuration", "angle_params", "beta"),
    [
        pytest.param(
            AxesConfiguration.Type1DA,
            {"delta": 9.0, "chi": 6.5, "chi0": 2.0, "xi": 5.25, "xi0": 1.75},
            4.5,
            id="type1da",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            {"delta": -5.0, "chi": -4.0, "chi0": 1.25, "xi": 6.75, "xi0": 0.75},
            -3.0,
            id="type2da",
        ),
    ],
)
def test_exact_da_cut_alpha_roundtrips_reference_cut(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
    beta: float,
    descending: bool,
) -> None:
    alpha = np.linspace(-15.0, 15.0, 61)
    if descending:
        alpha = alpha[::-1]

    kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha, beta, 24.0, **angle_params
    )
    slit_values = kx if configuration == AxesConfiguration.Type1DA else ky

    recovered = erlab.analysis.kspace.exact_cut_alpha(
        slit_values,
        beta,
        24.0,
        alpha,
        configuration,
        **angle_params,
    )

    assert np.allclose(recovered, alpha)


def test_exact_da_cut_alpha_returns_nan_out_of_domain() -> None:
    alpha_reference = np.linspace(-12.0, 12.0, 49)
    kx, _ = erlab.analysis.kspace.get_kconv_forward(AxesConfiguration.Type1DA)(
        alpha_reference,
        4.5,
        24.0,
        delta=9.0,
        chi=6.5,
        chi0=2.0,
        xi=5.25,
        xi0=1.75,
    )
    recovered = erlab.analysis.kspace.exact_cut_alpha(
        np.array([0.5 * float(kx.min() + kx.max()), float(kx.max()) + 0.1]),
        4.5,
        24.0,
        alpha_reference,
        AxesConfiguration.Type1DA,
        delta=9.0,
        chi=6.5,
        chi0=2.0,
        xi=5.25,
        xi0=1.75,
    )

    assert np.isfinite(recovered[0])
    assert np.isnan(recovered[1])


@pytest.mark.parametrize(
    ("configuration", "angle_params", "beta"),
    [
        pytest.param(
            AxesConfiguration.Type1DA,
            {"delta": 9.0, "chi": 6.5, "chi0": 2.0, "xi": 5.25, "xi0": 1.75},
            4.5,
            id="type1da",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            {"delta": -5.0, "chi": -4.0, "chi0": 1.25, "xi": 6.75, "xi0": 0.75},
            -3.0,
            id="type2da",
        ),
    ],
)
def test_exact_da_cut_alpha_accepts_scalar_and_ndarray_targets(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
    beta: float,
) -> None:
    alpha_reference = np.linspace(-15.0, 15.0, 61)
    kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha_reference, beta, 24.0, **angle_params
    )
    slit_values = kx if configuration == AxesConfiguration.Type1DA else ky

    scalar_index = 17
    scalar_recovered = erlab.analysis.kspace.exact_cut_alpha(
        float(slit_values[scalar_index]),
        beta,
        24.0,
        alpha_reference,
        configuration,
        **angle_params,
    )

    assert np.asarray(scalar_recovered).shape == ()
    assert np.isclose(
        float(np.asarray(scalar_recovered)), alpha_reference[scalar_index]
    )

    grid_index = np.array([[5, 11], [19, 27]])
    grid_targets = np.asarray(slit_values[grid_index], dtype=float)
    grid_recovered = erlab.analysis.kspace.exact_cut_alpha(
        grid_targets,
        beta,
        24.0,
        alpha_reference,
        configuration,
        **angle_params,
    )

    assert grid_recovered.shape == grid_targets.shape
    assert np.allclose(grid_recovered, alpha_reference[grid_index])


@pytest.mark.parametrize(
    ("configuration", "angle_params", "descending"),
    [
        pytest.param(
            AxesConfiguration.Type1DA,
            {"delta": 9.0, "chi": 6.5, "chi0": 2.0, "xi": 5.25, "xi0": 1.75},
            False,
            id="type1da-ascending",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            {"delta": -5.0, "chi": -4.0, "chi0": 1.25, "xi": 6.75, "xi0": 0.75},
            True,
            id="type2da-descending",
        ),
    ],
)
def test_exact_da_cut_alpha_broadcasts_over_beta_and_kinetic_energy(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
    descending: bool,
) -> None:
    alpha_reference = xr.DataArray(np.linspace(-15.0, 15.0, 61), dims=("alpha",))
    if descending:
        alpha_reference = alpha_reference[::-1]

    beta = xr.DataArray(
        np.array([-6.0, -1.0, 4.0]),
        dims=("hv",),
        coords={"hv": [20.0, 30.0, 40.0]},
    )
    kinetic_energy = xr.DataArray(
        np.array([[22.0, 22.5], [24.0, 24.5], [26.0, 26.5]]),
        dims=("hv", "eV"),
        coords={"hv": beta.hv, "eV": [-0.05, 0.05]},
    )

    kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha_reference,
        beta,
        kinetic_energy,
        **angle_params,
    )
    slit_axis = "kx" if configuration == AxesConfiguration.Type1DA else "ky"
    slit_values = (kx if configuration == AxesConfiguration.Type1DA else ky).rename(
        {"alpha": slit_axis}
    )

    recovered = erlab.analysis.kspace.exact_cut_alpha(
        slit_values,
        beta,
        kinetic_energy,
        alpha_reference,
        configuration,
        **angle_params,
    )
    expected = xr.broadcast(
        kinetic_energy, alpha_reference.rename({"alpha": slit_axis})
    )[1]

    xr.testing.assert_allclose(
        recovered.transpose("hv", "eV", slit_axis),
        expected.transpose("hv", "eV", slit_axis),
    )


def test_exact_da_cut_alpha_rejects_multivalued_reference_cut(monkeypatch) -> None:
    alpha_reference = xr.DataArray(np.linspace(-5.0, 5.0, 11), dims=("alpha",))

    def _mock_forward(alpha, beta, kinetic_energy, configuration, **kwargs):
        slit_source = xr.DataArray(
            np.array([0.0, 0.6, 1.0, 0.6, 0.0, -0.4, -0.8, -0.4, 0.0, 0.3, 0.1]),
            dims=("alpha",),
            coords={"alpha": alpha_reference.alpha},
        )
        zeros = xr.zeros_like(slit_source)
        if configuration == AxesConfiguration.Type1DA:
            return slit_source, zeros
        return zeros, slit_source

    def _mock_get_forward(configuration):
        def _forward(alpha, beta, kinetic_energy, **kwargs):
            return _mock_forward(alpha, beta, kinetic_energy, configuration, **kwargs)

        return _forward

    monkeypatch.setattr(erlab.analysis.kspace, "get_kconv_forward", _mock_get_forward)

    with pytest.raises(ValueError, match="non-physical angle value or offset"):
        erlab.analysis.kspace.exact_cut_alpha(
            np.array([0.1]),
            0.0,
            24.0,
            alpha_reference,
            AxesConfiguration.Type1DA,
        )


@pytest.mark.parametrize("descending", [False, True], ids=["ascending", "descending"])
def test_interp_monotonic_interpn_1d_handles_orientation_and_domain(
    descending: bool,
) -> None:
    source_coord = np.array([0.0, 1.0, 2.0])
    source_values = np.array([10.0, 20.0, 30.0])
    if descending:
        source_coord = source_coord[::-1]
        source_values = source_values[::-1]

    target = np.array([-1.0, 0.5, 1.5, 3.0])

    out = erlab.analysis.kspace._interp_monotonic_interpn_1d(
        source_coord, source_values, target
    )

    assert np.isnan(out[0])
    assert np.isclose(out[1], 15.0)
    assert np.isclose(out[2], 25.0)
    assert np.isnan(out[3])


def test_interp_monotonic_interpn_1d_rejects_non_monotonic() -> None:
    with pytest.raises(ValueError, match="non-physical angle value or offset"):
        erlab.analysis.kspace._interp_monotonic_interpn_1d(
            np.array([0.0, 2.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5]),
        )


def test_interp_monotonic_interpn_1d_rejects_constant_curve() -> None:
    with pytest.raises(ValueError, match="effectively constant"):
        erlab.analysis.kspace._interp_monotonic_interpn_1d(
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0]),
        )


def test_interp_monotonic_interpn_1d_rejects_repeated_monotonic_curve() -> None:
    with pytest.raises(ValueError, match="not strictly monotonic"):
        erlab.analysis.kspace._interp_monotonic_interpn_1d(
            np.array([0.0, 1.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([1.5]),
        )


def test_exact_cut_alpha_rejects_degenerate_dataarray_geometry(monkeypatch) -> None:
    alpha_reference = xr.DataArray(np.linspace(-5.0, 5.0, 5), dims=("alpha",))
    slit_target = xr.DataArray([0.0], dims=("kx",), coords={"kx": [0.0]})
    beta = xr.DataArray([0.0, 1.0], dims=("hv",), coords={"hv": [24.0, 32.0]})
    kinetic_energy = xr.DataArray([20.0, 24.0], dims=("hv",), coords={"hv": beta.hv})
    zeros = xr.zeros_like(beta)

    monkeypatch.setattr(
        erlab.analysis.kspace,
        "_fixed_beta_slit_coefficients",
        lambda *args, **kwargs: (zeros, zeros),
    )

    with pytest.raises(ValueError, match="slit-axis momentum does not vary with alpha"):
        erlab.analysis.kspace.exact_cut_alpha(
            slit_target,
            beta,
            kinetic_energy,
            alpha_reference,
            AxesConfiguration.Type1,
        )


@pytest.mark.parametrize(
    ("alpha_reference_r", "u_reference_norm", "match"),
    [
        pytest.param(
            np.zeros((2, 2)),
            np.zeros(4),
            "expects one-dimensional",
            id="ndim-mismatch",
        ),
        pytest.param(
            np.zeros(3),
            np.zeros(2),
            "matching shapes",
            id="shape-mismatch",
        ),
        pytest.param(
            np.zeros(1),
            np.zeros(1),
            "at least two alpha reference points",
            id="too-short",
        ),
    ],
)
def test_select_slit_branch_1d_validates_reference_inputs(
    alpha_reference_r: np.ndarray, u_reference_norm: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        erlab.analysis.kspace._select_slit_branch_1d(
            alpha_reference_r, u_reference_norm, 0.0
        )


def test_select_slit_branch_1d_rejects_effectively_constant_slit_curve() -> None:
    with pytest.raises(ValueError, match="effectively constant slit momentum"):
        erlab.analysis.kspace._select_slit_branch_1d(
            np.array([0.0, 1.0]),
            np.array([0.3, 0.3]),
            0.0,
        )


def test_select_slit_branch_1d_rejects_nonfinite_inverse_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        erlab.analysis.kspace,
        "_candidate_alpha_from_slit",
        lambda *args, **kwargs: np.array([np.nan, np.nan]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(
            ValueError, match="could not determine a valid inverse branch"
        ):
            erlab.analysis.kspace._select_slit_branch_1d(
                np.array([0.0, 1.0]),
                np.array([0.0, 0.1]),
                0.0,
            )


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(
            xr.DataArray(np.zeros((2, 2)), dims=("alpha", "hv")), id="dataarray"
        ),
        pytest.param(np.zeros((2, 2)), id="ndarray"),
    ],
)
def test_exact_1d_coord_requires_one_dimensional_inputs(values) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        erlab.analysis.kspace._exact_1d_coord(
            values,
            name="alpha_reference",
            default_dim="alpha",
            context="exact cut conversion",
        )


def test_exact_slit_target_coord_handles_alternate_dim_and_rejects_multidim() -> None:
    alt_dim_target = xr.DataArray([0.1, 0.2], dims=("momentum",))
    result, dim, is_dataarray = erlab.analysis.kspace._exact_slit_target_coord(
        alt_dim_target,
        slit_axis="kx",
        context="exact cut conversion",
    )

    assert result.identical(alt_dim_target)
    assert dim == "momentum"
    assert is_dataarray

    with pytest.raises(ValueError, match="one-dimensional or contain the slit-axis"):
        erlab.analysis.kspace._exact_slit_target_coord(
            xr.DataArray(np.zeros((2, 2)), dims=("row", "col")),
            slit_axis="kx",
            context="exact cut conversion",
        )


def test_exact_result_helpers_preserve_plain_arrays_and_skip_missing_dims() -> None:
    values = np.array([1.0, 2.0])
    assert (
        erlab.analysis.kspace._unwrap_exact_result(values, target_is_dataarray=False)
        is values
    )

    target = xr.DataArray(
        np.arange(6).reshape(2, 3),
        dims=("secondary", "primary"),
    )
    transposed = erlab.analysis.kspace._transpose_exact_target(
        target,
        primary_dim="primary",
        secondary_dim="missing",
        kinetic_energy=1.0,
        excluded_dim="hv",
    )

    assert transposed.dims == ("primary", "secondary")


@pytest.mark.parametrize(
    ("configuration", "angle_params", "beta_range"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            {"delta": 12.5, "xi": 7.25, "xi0": 2.5, "beta0": -3.75},
            (-10.0, -4.0),
            id="type1",
        ),
        pytest.param(
            AxesConfiguration.Type2,
            {"delta": -8.0, "xi": -6.5, "xi0": -1.75, "beta0": 2.25},
            (-8.0, -2.0),
            id="type2",
        ),
    ],
)
def test_exact_hv_slit_cut_coords_matches_manual_inversion(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
    beta_range: tuple[float, float],
) -> None:
    alpha_reference = xr.DataArray(np.linspace(-15.0, 15.0, 61), dims=("alpha",))
    hv = xr.DataArray(
        np.linspace(24.0, 60.0, 9),
        dims=("hv",),
        coords={"hv": np.linspace(24.0, 60.0, 9)},
    )
    eV = xr.DataArray(
        np.linspace(-0.2, 0.1, 4),
        dims=("eV",),
        coords={"eV": np.linspace(-0.2, 0.1, 4)},
    )
    beta = xr.DataArray(
        np.linspace(*beta_range, hv.size),
        dims=("hv",),
        coords={"hv": hv.values},
    )
    kinetic_energy = hv - 4.5 + eV

    slit_axis = "kx" if configuration == AxesConfiguration.Type1 else "ky"
    trial_alpha = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha_reference, beta, kinetic_energy, **angle_params
    )[0 if slit_axis == "kx" else 1]
    slit_min = float(trial_alpha.min())
    slit_max = float(trial_alpha.max())
    slit_value = xr.DataArray(
        np.linspace(slit_min * 0.6, slit_max * 0.6, 4),
        dims=(slit_axis,),
        coords={slit_axis: np.linspace(slit_min * 0.6, slit_max * 0.6, 4)},
    )

    alpha_source = erlab.analysis.kspace.exact_cut_alpha(
        slit_value,
        beta,
        kinetic_energy,
        alpha_reference,
        configuration,
        **angle_params,
    )
    other_momentum = erlab.analysis.kspace._fixed_beta_other_axis_momentum(
        alpha_source,
        beta,
        kinetic_energy,
        configuration,
        **angle_params,
    )

    if configuration == AxesConfiguration.Type1:
        kx_source = slit_value
        ky_source = other_momentum
    else:
        kx_source = other_momentum
        ky_source = slit_value

    kz_source = erlab.analysis.kspace.kz_func(
        kinetic_energy, 10.0, kx_source, ky_source
    )
    kz_min = float(kz_source.min())
    kz_max = float(kz_source.max())
    kz_value = xr.DataArray(
        np.linspace(
            kz_min + 0.1 * (kz_max - kz_min), kz_max - 0.1 * (kz_max - kz_min), 7
        ),
        dims=("kz",),
        coords={
            "kz": np.linspace(
                kz_min + 0.1 * (kz_max - kz_min), kz_max - 0.1 * (kz_max - kz_min), 7
            )
        },
    )

    alpha_target, hv_target, _ = erlab.analysis.kspace.exact_hv_cut_coords(
        slit_value,
        kz_value,
        beta,
        hv,
        kinetic_energy,
        alpha_reference,
        configuration,
        10.0,
        **angle_params,
    )

    expected_alpha = np.empty((slit_value.size, kz_value.size, eV.size), dtype=float)
    expected_hv = np.empty_like(expected_alpha)
    kz_by_slice = kz_source.transpose(slit_axis, "eV", "hv").values
    alpha_by_slice = alpha_source.transpose(slit_axis, "eV", "hv").values

    for i in range(slit_value.size):
        for j in range(eV.size):
            expected_alpha[i, :, j] = _manual_monotonic_interp(
                kz_by_slice[i, j], alpha_by_slice[i, j], kz_value.values
            )
            expected_hv[i, :, j] = _manual_monotonic_interp(
                kz_by_slice[i, j], hv.values, kz_value.values
            )

    expected_alpha_da = xr.DataArray(
        expected_alpha,
        dims=(slit_axis, "kz", "eV"),
        coords={slit_axis: slit_value.values, "kz": kz_value.values, "eV": eV.values},
    )
    expected_hv_da = xr.DataArray(
        expected_hv,
        dims=(slit_axis, "kz", "eV"),
        coords={slit_axis: slit_value.values, "kz": kz_value.values, "eV": eV.values},
    )

    xr.testing.assert_allclose(alpha_target, expected_alpha_da)
    xr.testing.assert_allclose(hv_target, expected_hv_da)


def test_exact_hv_slit_cut_coords_accepts_scalar_slit_target() -> None:
    alpha_reference = xr.DataArray(np.linspace(-15.0, 15.0, 61), dims=("alpha",))
    hv = xr.DataArray(
        np.linspace(24.0, 60.0, 9),
        dims=("hv",),
        coords={"hv": np.linspace(24.0, 60.0, 9)},
    )
    eV = xr.DataArray(
        np.linspace(-0.2, 0.1, 4),
        dims=("eV",),
        coords={"eV": np.linspace(-0.2, 0.1, 4)},
    )
    beta = xr.DataArray(
        np.linspace(-10.0, -4.0, hv.size),
        dims=("hv",),
        coords={"hv": hv.values},
    )
    kinetic_energy = hv - 4.5 + eV
    angle_params = {"delta": 12.5, "xi": 7.25, "xi0": 2.5, "beta0": -3.75}

    slit_values = erlab.analysis.kspace.get_kconv_forward(AxesConfiguration.Type1)(
        alpha_reference, beta, kinetic_energy, **angle_params
    )[0]
    slit_target = float(
        np.linspace(float(slit_values.min()) * 0.6, float(slit_values.max()) * 0.6, 4)[
            2
        ]
    )

    alpha_source = erlab.analysis.kspace.exact_cut_alpha(
        xr.DataArray([slit_target], dims=("kx",), coords={"kx": [slit_target]}),
        beta,
        kinetic_energy,
        alpha_reference,
        AxesConfiguration.Type1,
        **angle_params,
    )
    other_momentum = erlab.analysis.kspace._fixed_beta_other_axis_momentum(
        alpha_source,
        beta,
        kinetic_energy,
        AxesConfiguration.Type1,
        **angle_params,
    )
    kz_source = erlab.analysis.kspace.kz_func(
        kinetic_energy, 10.0, slit_target, other_momentum
    )
    kz_value = xr.DataArray(
        np.linspace(
            float(kz_source.min()) + 0.1 * float(kz_source.max() - kz_source.min()),
            float(kz_source.max()) - 0.1 * float(kz_source.max() - kz_source.min()),
            7,
        ),
        dims=("kz",),
        coords={
            "kz": np.linspace(
                float(kz_source.min()) + 0.1 * float(kz_source.max() - kz_source.min()),
                float(kz_source.max()) - 0.1 * float(kz_source.max() - kz_source.min()),
                7,
            )
        },
    )

    alpha_scalar, hv_scalar, other_scalar = erlab.analysis.kspace.exact_hv_cut_coords(
        slit_target,
        kz_value,
        beta,
        hv,
        kinetic_energy,
        alpha_reference,
        AxesConfiguration.Type1,
        10.0,
        **angle_params,
    )
    alpha_vector, hv_vector, other_vector = erlab.analysis.kspace.exact_hv_cut_coords(
        xr.DataArray([slit_target], dims=("kx",), coords={"kx": [slit_target]}),
        kz_value,
        beta,
        hv,
        kinetic_energy,
        alpha_reference,
        AxesConfiguration.Type1,
        10.0,
        **angle_params,
    )

    xr.testing.assert_allclose(alpha_scalar, alpha_vector.squeeze("kx", drop=True))
    xr.testing.assert_allclose(hv_scalar, hv_vector.squeeze("kx", drop=True))
    xr.testing.assert_allclose(other_scalar, other_vector.squeeze("kx", drop=True))


@pytest.mark.parametrize(
    ("configuration", "angle_params", "beta_range"),
    [
        pytest.param(
            AxesConfiguration.Type1DA,
            {"delta": 9.0, "chi": 6.5, "chi0": 2.0, "xi": 5.25, "xi0": 1.75},
            (-6.0, 2.0),
            id="type1da",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            {"delta": -5.0, "chi": -4.0, "chi0": 1.25, "xi": 6.75, "xi0": 0.75},
            (-4.0, 4.0),
            id="type2da",
        ),
    ],
)
def test_exact_hv_da_cut_coords_matches_manual_inversion(
    configuration: AxesConfiguration,
    angle_params: dict[str, float],
    beta_range: tuple[float, float],
) -> None:
    alpha_reference = xr.DataArray(np.linspace(-15.0, 15.0, 61), dims=("alpha",))
    hv = xr.DataArray(
        np.linspace(24.0, 54.0, 7),
        dims=("hv",),
        coords={"hv": np.linspace(24.0, 54.0, 7)},
    )
    eV = xr.DataArray(
        np.linspace(-0.15, 0.1, 3),
        dims=("eV",),
        coords={"eV": np.linspace(-0.15, 0.1, 3)},
    )
    beta = xr.DataArray(
        np.linspace(*beta_range, hv.size),
        dims=("hv",),
        coords={"hv": hv.values},
    )
    kinetic_energy = hv - 4.5 + eV

    slit_axis = "kx" if configuration == AxesConfiguration.Type1DA else "ky"
    trial_slit = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha_reference, beta, kinetic_energy, **angle_params
    )[0 if slit_axis == "kx" else 1]
    slit_value = xr.DataArray(
        np.linspace(float(trial_slit.min()) * 0.6, float(trial_slit.max()) * 0.6, 4),
        dims=(slit_axis,),
        coords={
            slit_axis: np.linspace(
                float(trial_slit.min()) * 0.6, float(trial_slit.max()) * 0.6, 4
            )
        },
    )

    alpha_source = erlab.analysis.kspace.exact_cut_alpha(
        slit_value,
        beta,
        kinetic_energy,
        alpha_reference,
        configuration,
        **angle_params,
    )
    other_momentum = erlab.analysis.kspace._exact_other_axis_momentum(
        alpha_source,
        beta,
        kinetic_energy,
        configuration,
        **angle_params,
    )

    if configuration == AxesConfiguration.Type1DA:
        kx_source = slit_value
        ky_source = other_momentum
    else:
        kx_source = other_momentum
        ky_source = slit_value

    kz_source = erlab.analysis.kspace.kz_func(
        kinetic_energy, 10.0, kx_source, ky_source
    )
    kz_value = xr.DataArray(
        np.linspace(float(kz_source.min()) + 0.05, float(kz_source.max()) - 0.05, 7),
        dims=("kz",),
        coords={
            "kz": np.linspace(
                float(kz_source.min()) + 0.05, float(kz_source.max()) - 0.05, 7
            )
        },
    )

    alpha_target, hv_target, _ = erlab.analysis.kspace.exact_hv_cut_coords(
        slit_value,
        kz_value,
        beta,
        hv,
        kinetic_energy,
        alpha_reference,
        configuration,
        10.0,
        **angle_params,
    )

    expected_alpha = np.empty((slit_value.size, kz_value.size, eV.size), dtype=float)
    expected_hv = np.empty_like(expected_alpha)
    kz_by_slice = kz_source.transpose(slit_axis, "eV", "hv").values
    alpha_by_slice = alpha_source.transpose(slit_axis, "eV", "hv").values

    for i in range(slit_value.size):
        for j in range(eV.size):
            expected_alpha[i, :, j] = _manual_monotonic_interp(
                kz_by_slice[i, j], alpha_by_slice[i, j], kz_value.values
            )
            expected_hv[i, :, j] = _manual_monotonic_interp(
                kz_by_slice[i, j], hv.values, kz_value.values
            )

    expected_alpha_da = xr.DataArray(
        expected_alpha,
        dims=(slit_axis, "kz", "eV"),
        coords={slit_axis: slit_value.values, "kz": kz_value.values, "eV": eV.values},
    )
    expected_hv_da = xr.DataArray(
        expected_hv,
        dims=(slit_axis, "kz", "eV"),
        coords={slit_axis: slit_value.values, "kz": kz_value.values, "eV": eV.values},
    )

    xr.testing.assert_allclose(alpha_target, expected_alpha_da)
    xr.testing.assert_allclose(hv_target, expected_hv_da)
