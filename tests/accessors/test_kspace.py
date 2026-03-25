import re

import numpy as np
import pytest
import scipy.optimize
import xarray
import xarray.testing

import erlab.analysis.kspace
from erlab.accessors.kspace import IncompleteDataError
from erlab.constants import AxesConfiguration
from erlab.io.exampledata import generate_hvdep_cuts


@pytest.fixture(scope="module")
def hvdep():
    data = generate_hvdep_cuts((50, 250, 300), seed=1)
    data.kspace.inner_potential = 10.0
    return data


@pytest.fixture(scope="module")
def config_1_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.attrs["configuration"] = 1
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_2_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.attrs["configuration"] = 2
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_3_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.attrs["configuration"] = 3
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_4_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.attrs["configuration"] = 4
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_1_cut(cut):
    data = cut.copy(deep=True)
    data.attrs["configuration"] = 1
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_2_cut(cut):
    data = cut.copy(deep=True)
    data.attrs["configuration"] = 2
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_3_cut(cut):
    data = cut.copy(deep=True)
    data.attrs["configuration"] = 3
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_4_cut(cut):
    data = cut.copy(deep=True)
    data.attrs["configuration"] = 4
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_1_hvdep(hvdep):
    data = hvdep.copy(deep=True)
    data.attrs["configuration"] = 1
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_2_hvdep(hvdep):
    data = hvdep.copy(deep=True)
    data.attrs["configuration"] = 2
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_3_hvdep(hvdep):
    data = hvdep.copy(deep=True)
    data.attrs["configuration"] = 3
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_4_hvdep(hvdep):
    data = hvdep.copy(deep=True)
    data.attrs["configuration"] = 4
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


_NORMAL_EMISSION_CASES = [
    pytest.param(
        AxesConfiguration.Type1,
        {"xi": 7.25},
        {"delta": 12.5, "xi": 2.5, "beta": -3.75},
        [4.75, -3.75],
        id="Type1",
    ),
    pytest.param(
        AxesConfiguration.Type2,
        {"xi": -6.5},
        {"delta": -8.0, "xi": -1.75, "beta": 2.25},
        [-4.75, 2.25],
        id="Type2",
    ),
    pytest.param(
        AxesConfiguration.Type1DA,
        {"xi": 5.25, "chi": 6.5},
        {"delta": 9.0, "chi": 2.0, "xi": 1.75},
        [3.5, -4.5],
        id="Type1DA",
    ),
    pytest.param(
        AxesConfiguration.Type2DA,
        {"xi": 6.75, "chi": -4.0},
        {"delta": -7.5, "chi": 1.25, "xi": 2.5},
        [-5.25, 4.25],
        id="Type2DA",
    ),
]


def _make_normal_emission_data(
    anglemap,
    configuration: AxesConfiguration,
    coords: dict[str, float],
) -> xarray.DataArray:
    data = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(0, 3)).copy(
        deep=True
    )
    data.attrs["configuration"] = int(configuration)
    return data.assign_coords(**coords)


def _solve_normal_emission_angles(
    configuration: AxesConfiguration,
    coords: dict[str, float],
    offsets: dict[str, float],
    initial_guess: list[float],
) -> tuple[float, float]:
    forward = erlab.analysis.kspace.get_kconv_forward(configuration)

    if configuration in (AxesConfiguration.Type1, AxesConfiguration.Type2):
        angle_params = {
            "delta": offsets["delta"],
            "xi": coords["xi"],
            "xi0": offsets["xi"],
            "beta0": offsets["beta"],
        }
    else:
        angle_params = {
            "delta": offsets["delta"],
            "chi": coords["chi"],
            "chi0": offsets["chi"],
            "xi": coords["xi"],
            "xi0": offsets["xi"],
        }

    result = scipy.optimize.root(
        lambda angles: np.array(
            [float(v) for v in forward(angles[0], angles[1], 1.0, **angle_params)]
        ),
        x0=np.asarray(initial_guess, dtype=float),
    )

    assert result.success, result.message
    return float(result.x[0]), float(result.x[1])


@pytest.mark.parametrize("data_type", ["anglemap", "cut", "hvdep"])
def test_offsets(data_type, request) -> None:
    data = request.getfixturevalue(data_type).copy(deep=True)
    data.kspace.offsets.reset()
    data.kspace.offsets = {"xi": 10.0}
    answer = dict.fromkeys(data.kspace._valid_offset_keys, 0.0)
    answer["xi"] = 10.0
    assert dict(data.kspace.offsets) == answer

    with pytest.raises(
        KeyError,
        match=re.escape(
            "Invalid offset key 'invalid' for experimental configuration "
            f"{data.kspace.configuration}. Valid keys are: "
            f"{data.kspace._valid_offset_keys}."
        ),
    ):
        data.kspace.offsets["invalid"] = 10.0


@pytest.mark.parametrize(
    ("configuration", "coords", "true_offsets", "initial_guess"),
    _NORMAL_EMISSION_CASES,
)
def test_offsets_repr_shows_normal_emission(
    anglemap,
    configuration: AxesConfiguration,
    coords: dict[str, float],
    true_offsets: dict[str, float],
    initial_guess: list[float],
) -> None:
    data = _make_normal_emission_data(anglemap, configuration, coords)
    data.kspace.offsets = true_offsets
    alpha_normal, beta_normal = _solve_normal_emission_angles(
        configuration, coords, true_offsets, initial_guess
    )

    repr_str = repr(data.kspace.offsets)
    repr_match = re.search(r"normal emission: alpha=([^,]+), beta=([^\n]+)", repr_str)
    assert repr_match is not None
    assert float(repr_match.group(1)) == pytest.approx(alpha_normal, abs=1e-10)
    assert float(repr_match.group(2)) == pytest.approx(beta_normal, abs=1e-10)

    html = data.kspace.offsets._repr_html_()
    assert "normal alpha" in html
    assert "normal beta" in html

    html_alpha = re.search(r"<th>normal alpha</th><td>([^<]+)</td>", html)
    html_beta = re.search(r"<th>normal beta</th><td>([^<]+)</td>", html)
    assert html_alpha is not None
    assert html_beta is not None
    assert float(html_alpha.group(1)) == pytest.approx(alpha_normal, abs=1e-10)
    assert float(html_beta.group(1)) == pytest.approx(beta_normal, abs=1e-10)


def test_offsets_repr_omits_normal_emission_without_required_coords(anglemap) -> None:
    data = _make_normal_emission_data(anglemap, AxesConfiguration.Type1, {})
    data = data.reset_coords("xi", drop=True)

    assert repr(data.kspace.offsets) == "{'delta': 0.0, 'xi': 0.0, 'beta': 0.0}"

    html = data.kspace.offsets._repr_html_()
    assert "normal alpha" not in html
    assert "normal beta" not in html


def test_offsets_repr_normalizes_zero_normal_emission(anglemap) -> None:
    data = _make_normal_emission_data(
        anglemap, AxesConfiguration.Type2DA, {"xi": 0.0, "chi": 0.0}
    )
    data.kspace.offsets = {"delta": 0.0, "chi": 0.0, "xi": 0.0}

    assert "normal emission: alpha=0.0, beta=0.0" in repr(data.kspace.offsets)

    html = data.kspace.offsets._repr_html_()
    assert "<th>normal alpha</th><td>0.0</td>" in html
    assert "<th>normal beta</th><td>0.0</td>" in html


@pytest.mark.parametrize(
    ("configuration", "coords", "true_offsets", "initial_guess"),
    _NORMAL_EMISSION_CASES,
)
def test_set_normal(
    anglemap,
    configuration: AxesConfiguration,
    coords: dict[str, float],
    true_offsets: dict[str, float],
    initial_guess: list[float],
) -> None:
    data = _make_normal_emission_data(anglemap, configuration, coords)
    alpha_normal, beta_normal = _solve_normal_emission_angles(
        configuration, coords, true_offsets, initial_guess
    )

    data.kspace.set_normal(alpha_normal, beta_normal, delta=true_offsets["delta"])

    for key, expected in true_offsets.items():
        assert data.kspace.offsets[key] == pytest.approx(expected)

    kx, ky = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha_normal, beta_normal, 1.0, **data.kspace.angle_params
    )

    assert float(kx) == pytest.approx(0.0, abs=1e-10)
    assert float(ky) == pytest.approx(0.0, abs=1e-10)


def test_set_normal_preserves_delta_when_omitted(
    anglemap,
) -> None:
    configuration = AxesConfiguration.Type2
    coords = {"xi": 6.5}
    reference_offsets = {"delta": 3.0, "xi": 1.25, "beta": -2.75}
    alpha_normal, beta_normal = _solve_normal_emission_angles(
        configuration,
        coords,
        reference_offsets,
        [coords["xi"] - reference_offsets["xi"], reference_offsets["beta"]],
    )

    data = _make_normal_emission_data(anglemap, configuration, coords)
    data.kspace.offsets = {"delta": -17.5, "xi": 0.0, "beta": 0.0}

    data.kspace.set_normal(alpha_normal, beta_normal)

    assert data.kspace.offsets["delta"] == pytest.approx(-17.5)
    assert data.kspace.offsets["xi"] == pytest.approx(reference_offsets["xi"])
    assert data.kspace.offsets["beta"] == pytest.approx(reference_offsets["beta"])


def test_set_normal_overwrites_delta(anglemap) -> None:
    configuration = AxesConfiguration.Type2DA
    coords = {"xi": 4.5, "chi": -3.0}
    reference_offsets = {"delta": -5.0, "chi": 1.25, "xi": 0.75}
    alpha_normal, beta_normal = _solve_normal_emission_angles(
        configuration,
        coords,
        reference_offsets,
        [
            coords["chi"] - reference_offsets["chi"],
            coords["xi"] - reference_offsets["xi"],
        ],
    )

    data = _make_normal_emission_data(anglemap, configuration, coords)
    data.kspace.offsets = {"delta": 11.0, "chi": 0.0, "xi": 0.0}

    data.kspace.set_normal(alpha_normal, beta_normal, delta=8.5)

    assert data.kspace.offsets["delta"] == pytest.approx(8.5)
    assert data.kspace.offsets["chi"] == pytest.approx(reference_offsets["chi"])
    assert data.kspace.offsets["xi"] == pytest.approx(reference_offsets["xi"])


@pytest.mark.parametrize(
    ("target_configuration", "target_coords"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            {"xi": 7.25},
            id="Type1",
        ),
        pytest.param(
            AxesConfiguration.Type2,
            {"xi": -6.5},
            id="Type2",
        ),
        pytest.param(
            AxesConfiguration.Type1DA,
            {"xi": 5.25, "chi": 6.5},
            id="Type1DA",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            {"xi": 6.75, "chi": -4.0},
            id="Type2DA",
        ),
    ],
)
def test_set_normal_like(
    anglemap,
    target_configuration: AxesConfiguration,
    target_coords: dict[str, float],
) -> None:
    source_configuration = AxesConfiguration.Type1DA
    source_coords = {"xi": 5.25, "chi": 6.5}
    source_offsets = {"delta": 9.0, "chi": 2.0, "xi": 1.75}

    source = _make_normal_emission_data(anglemap, source_configuration, source_coords)
    source.kspace.offsets = source_offsets
    alpha_normal, beta_normal = _solve_normal_emission_angles(
        source_configuration, source_coords, source_offsets, [3.5, -4.5]
    )

    target = _make_normal_emission_data(anglemap, target_configuration, target_coords)
    target.kspace.set_normal_like(source)

    expected_offsets = erlab.analysis.kspace._offsets_from_normal_emission(
        target_configuration,
        alpha_normal,
        beta_normal,
        xi=target_coords["xi"],
        chi=target_coords.get("chi"),
    )
    expected_offsets["delta"] = source_offsets["delta"]

    for key, expected in expected_offsets.items():
        assert target.kspace.offsets[key] == pytest.approx(expected)

    kx, ky = erlab.analysis.kspace.get_kconv_forward(target_configuration)(
        alpha_normal, beta_normal, 1.0, **target.kspace.angle_params
    )

    assert float(kx) == pytest.approx(0.0, abs=1e-10)
    assert float(ky) == pytest.approx(0.0, abs=1e-10)


def test_set_normal_like_rejects_non_dataarray(anglemap) -> None:
    with pytest.raises(TypeError, match=r"`other` must be an xarray\.DataArray\."):
        anglemap.kspace.set_normal_like(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(("alpha", "beta"), [(np.nan, 0.0), (0.0, np.inf)])
def test_set_normal_rejects_nonfinite_input(
    anglemap, alpha: float, beta: float
) -> None:
    data = _make_normal_emission_data(
        anglemap, AxesConfiguration.Type1, {"xi": float(anglemap.xi.values)}
    )

    with pytest.raises(ValueError, match=r"must be a finite scalar"):
        data.kspace.set_normal(alpha, beta)


def test_set_normal_missing_chi(anglemap) -> None:
    data = _make_normal_emission_data(
        anglemap, AxesConfiguration.Type2DA, {"xi": float(anglemap.xi.values)}
    )

    with pytest.raises(
        IncompleteDataError,
        match=IncompleteDataError._make_message("coord", "chi"),
    ):
        data.kspace.set_normal(0.0, 0.0)


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no-dask"])
@pytest.mark.parametrize("energy_axis", ["kinetic", "binding"])
@pytest.mark.parametrize("data_type", ["anglemap", "cut", "hvdep"])
@pytest.mark.parametrize(
    "configuration", AxesConfiguration, ids=[c.name for c in AxesConfiguration]
)
@pytest.mark.parametrize("extra_dims", [0, 1, 2], ids=["extra0", "extra1", "extra2"])
def test_kconv(
    use_dask: bool,
    energy_axis: str,
    data_type: str,
    configuration: AxesConfiguration,
    extra_dims: int,
    request,
) -> None:
    data = request.getfixturevalue(data_type).copy(deep=True)

    expected = request.getfixturevalue(
        f"config_{configuration.value}_{data_type.replace('angle', 'k')}"
    )

    if energy_axis == "kinetic":
        data = data.assign_coords(eV=data.hv - data.kspace.work_function + data.eV)

    if extra_dims > 0:
        data = data.expand_dims({f"extra{i}": 2 for i in range(extra_dims)})

    if use_dask:
        data = data.chunk()

    data.attrs["configuration"] = configuration.value

    for c in AxesConfiguration:
        data_ = data.kspace.as_configuration(c)
        if c == configuration:
            xarray.testing.assert_identical(data_, data)
        else:
            assert data_.attrs["configuration"] == c.value

    del data_

    match configuration:
        case AxesConfiguration.Type1DA | AxesConfiguration.Type2DA:
            data = data.assign_coords(chi=0.0)

    if data_type == "hvdep":
        data.kspace.inner_potential = 10.0

    if energy_axis == "kinetic":
        if data_type == "hvdep":
            with pytest.raises(
                ValueError,
                match=r"Energy axis of photon energy dependent data must be in "
                r"binding energy.",
            ):
                kconv = data.kspace.convert(silent=False)
            return

        with pytest.warns(
            UserWarning,
            match="The energy axis seems to be in terms of kinetic energy, "
            "attempting conversion to binding energy.",
        ):
            kconv = data.kspace.convert(silent=False)
    else:
        kconv = data.kspace.convert(silent=False)

    if use_dask:
        kconv = kconv.compute()

    if extra_dims > 0:
        for j in range(1, extra_dims):
            xarray.testing.assert_allclose(
                expected, kconv.isel({f"extra{i}": j for i in range(extra_dims)})
            )
    else:
        xarray.testing.assert_allclose(expected, kconv)

    if data_type == "anglemap":
        assert len(kconv.shape) == 3 + extra_dims
        if extra_dims == 0:
            assert set(kconv.shape) == {10, 310}
        else:
            assert set(kconv.shape) == {10, 310, 2}

    elif data_type == "cut":
        assert len(kconv.shape) == 2 + extra_dims
        if extra_dims == 0:
            assert set(kconv.shape) == {310, 500}
        else:
            assert set(kconv.shape) == {310, 500, 2}

    elif data_type == "hvdep":
        assert len(kconv.shape) == 3 + extra_dims
        if extra_dims == 0:
            assert set(kconv.shape) == {637, 63, 300}
        else:
            assert set(kconv.shape) == {637, 63, 300, 2}

    assert isinstance(kconv, xarray.DataArray)
    assert not kconv.isnull().all()


@pytest.mark.parametrize("missing_coord", ["alpha", "beta", "chi", "xi", "eV", "hv"])
def test_kconv_missing_coord(missing_coord, anglemap):
    data = anglemap.copy().assign_coords(chi=0.0)
    data.attrs["configuration"] = 4

    data = data.drop_vars(missing_coord)

    with pytest.raises(
        IncompleteDataError,
        match=IncompleteDataError._make_message("coord", missing_coord),
    ):
        data.kspace.convert(silent=True)


@pytest.mark.parametrize("missing_attr", ["configuration"])
def test_kconv_missing_attr(missing_attr, anglemap):
    data = anglemap.copy().assign_coords(chi=0.0)
    data.attrs["configuration"] = 4
    data.attrs.pop(missing_attr)

    with pytest.raises(
        IncompleteDataError,
        match=IncompleteDataError._make_message("attr", missing_attr),
    ):
        data.kspace.convert(silent=True)


def test_kspace_set_existing_configuration(anglemap):
    data = anglemap.copy().assign_coords(chi=0.0)

    with pytest.raises(
        AttributeError,
        match=r"Configuration is already set. To modify the experimental "
        r"configuration, use `DataArray.kspace.as_configuration`.",
    ):
        data.kspace.configuration = 4


def test_resolution_raises_nonpositive_kinetic_energy(anglemap) -> None:
    data = anglemap.copy(deep=True)
    data.kspace.work_function = 49.7

    with pytest.raises(
        ValueError,
        match=r"Nonphysical kinetic energy detected while estimating in-plane "
        r"momentum resolution: min\(E_k\)=",
    ):
        _ = data.kspace.best_kp_resolution

    with pytest.raises(
        ValueError,
        match=r"Nonphysical kinetic energy detected while estimating out-of-plane "
        r"momentum resolution: min\(E_k\)=",
    ):
        _ = data.kspace.best_kz_resolution

    with pytest.raises(
        ValueError,
        match=r"Nonphysical kinetic energy detected while converting to momentum "
        r"space: min\(E_k\)=",
    ):
        _ = data.kspace.convert(silent=True)


def test_estimate_resolution_from_numpoints_kz_uses_adjacent_spacing(hvdep) -> None:
    data = hvdep.copy(deep=True)
    lims = data.kspace.estimate_bounds()["kz"]

    out = data.kspace.estimate_resolution("kz", lims=lims, from_numpoints=True)
    expected = float((lims[1] - lims[0]) / (data.hv.size - 1))

    assert out == pytest.approx(expected)


def test_estimate_resolution_from_numpoints_single_point_returns_inf(anglemap) -> None:
    data = anglemap.isel(alpha=slice(0, 1)).copy(deep=True)
    axis = data.kspace.slit_axis

    out = data.kspace.estimate_resolution(axis, lims=(0.0, 1.0), from_numpoints=True)

    assert np.isinf(out)


def test_finite_minmax_all_nonfinite(anglemap) -> None:
    data = anglemap.copy(deep=True)
    mn, mx = data.kspace._finite_minmax(np.array([np.nan, np.inf, -np.inf]))
    assert np.isnan(mn)
    assert np.isnan(mx)


def test_check_kinetic_energy_warns_without_raising_for_nonfinite(anglemap) -> None:
    data = anglemap.copy(deep=True)
    kinetic = np.array([np.nan, np.inf, -np.inf])
    with pytest.warns(
        UserWarning,
        match=r"Cannot proceed while test context: kinetic energy contains no finite "
        r"values.",
    ):
        result = data.kspace._check_kinetic_energy(
            context="test context",
            kinetic_energy=kinetic,
            raise_on_violation=False,
        )
    assert result is kinetic


def test_check_kinetic_energy_raises_for_nonfinite(anglemap) -> None:
    data = anglemap.copy(deep=True)
    with pytest.raises(
        ValueError,
        match=r"Cannot proceed while test context: kinetic energy contains no finite "
        r"values.",
    ):
        _ = data.kspace._check_kinetic_energy(
            context="test context",
            kinetic_energy=np.array([np.nan, np.inf, -np.inf]),
        )


def test_check_kinetic_energy_warns_without_raising_for_nonphysical(anglemap) -> None:
    data = anglemap.copy(deep=True)
    data.kspace.work_function = 49.7
    with pytest.warns(
        UserWarning,
        match=r"Nonphysical kinetic energy detected while test context: min\(E_k\)=",
    ):
        result = data.kspace._check_kinetic_energy(
            context="test context",
            raise_on_violation=False,
        )
    xarray.testing.assert_identical(result, data.kspace._kinetic_energy)


def test_convert_coords_assigns_momentum_coords(anglemap) -> None:
    data = anglemap.copy(deep=True)
    out = data.kspace.convert_coords()

    assert "kx" in out.coords
    assert "ky" in out.coords
    assert set(out.kx.dims) == set(data.dims)
    assert set(out.ky.dims) == set(data.dims)
    assert np.isfinite(out.kx.values).all()
    assert np.isfinite(out.ky.values).all()


def test_hv_to_kz_accepts_iterable_hv(config_1_hvdep) -> None:
    out = config_1_hvdep.kspace.hv_to_kz([30.0, 45.0, 60.0])

    assert "hv" in out.dims
    assert out.hv.size == 3
    assert np.isfinite(out.values).all()
