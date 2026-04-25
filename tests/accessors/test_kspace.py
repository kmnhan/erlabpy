import functools
import re

import numpy as np
import pytest
import scipy.optimize
import xarray
import xarray.testing

import erlab.accessors.kspace
import erlab.analysis.kspace
from erlab.accessors.kspace import IncompleteDataError
from erlab.constants import AxesConfiguration
from erlab.io.exampledata import generate_hvdep_cuts


@pytest.fixture(scope="module")
def hvdep():
    data = generate_hvdep_cuts((24, 120, 140), seed=1)
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


def _make_alpha_field_cut(
    cut: xarray.DataArray,
    configuration: AxesConfiguration,
    *,
    xi: float,
    offsets: dict[str, float],
    beta: float | None = None,
    chi: float | None = None,
) -> xarray.DataArray:
    data = cut.copy(deep=True)
    data.attrs["configuration"] = int(configuration)
    data = data.assign_coords(xi=xi)
    if chi is not None:
        data = data.assign_coords(chi=chi)
    if beta is not None:
        data = data.assign_coords(beta=beta)

    alpha_grid = xarray.broadcast(data.eV, data.alpha)[1].transpose(*data.dims)
    data = data.copy(data=alpha_grid.values.astype(float))
    data.kspace.offsets = offsets
    return data


def _exact_cut_target_alpha(
    data: xarray.DataArray, slit_momentum: np.ndarray
) -> xarray.DataArray:
    slit_axis = data.kspace.slit_axis
    slit_value = xarray.DataArray(
        slit_momentum, dims=slit_axis, coords={slit_axis: slit_momentum}
    )
    return erlab.analysis.kspace.exact_cut_alpha(
        slit_value,
        data.beta,
        data.kspace._kinetic_energy,
        data.alpha,
        data.kspace.configuration,
        **data.kspace.angle_params,
    )


def _legacy_cut_target_alpha(
    data: xarray.DataArray, slit_momentum: np.ndarray
) -> xarray.DataArray:
    slit_axis = data.kspace.slit_axis
    other_axis = data.kspace.other_axis
    other_lims = data.kspace.estimate_bounds()[other_axis]
    other_mean = np.array([(other_lims[0] + other_lims[1]) / 2.0])

    slit_value = xarray.DataArray(
        slit_momentum, dims=slit_axis, coords={slit_axis: slit_momentum}
    )
    other_value = xarray.DataArray(
        other_mean, dims=other_axis, coords={other_axis: other_mean}
    )

    if slit_axis == "kx":
        alpha, _ = data.kspace._inverse_func(slit_value, other_value)
    else:
        alpha, _ = data.kspace._inverse_func(other_value, slit_value)

    return alpha.squeeze(other_axis, drop=True)


def _mask_alpha_domain(
    data: xarray.DataArray, target_alpha: xarray.DataArray
) -> xarray.DataArray:
    alpha_min = float(data.alpha.min())
    alpha_max = float(data.alpha.max())
    tol = 1e-9
    valid = (target_alpha >= alpha_min - tol) & (target_alpha <= alpha_max + tol)
    return target_alpha.where(valid)


def _make_field_hvdep_cut(
    hvdep: xarray.DataArray,
    configuration: AxesConfiguration,
    *,
    field: str,
    xi: float,
    offsets: dict[str, float],
    chi: float | None = None,
) -> xarray.DataArray:
    data = hvdep.copy(deep=True)
    data.attrs["configuration"] = int(configuration)
    data = data.assign_coords(xi=xi)
    if chi is not None:
        data = data.assign_coords(chi=chi)

    alpha_grid, _, hv_grid = xarray.broadcast(data.alpha, data.eV, data.hv)
    match field:
        case "alpha":
            values = alpha_grid
        case "hv":
            values = hv_grid
        case _:
            raise ValueError(f"Unsupported field '{field}'")

    data = data.copy(data=values.transpose(*data.dims).values.astype(float))
    data.kspace.offsets = offsets
    data.kspace.inner_potential = 10.0
    return data


def _exact_hvdep_targets(
    data: xarray.DataArray, slit_momentum: np.ndarray, kz: np.ndarray
) -> tuple[xarray.DataArray, xarray.DataArray]:
    slit_axis = data.kspace.slit_axis
    slit_value = xarray.DataArray(
        slit_momentum, dims=slit_axis, coords={slit_axis: slit_momentum}
    )
    kz_value = xarray.DataArray(kz, dims="kz", coords={"kz": kz})
    alpha, hv, _ = erlab.analysis.kspace.exact_hv_cut_coords(
        slit_value,
        kz_value,
        data.beta,
        data.hv,
        data.kspace._kinetic_energy,
        data.alpha,
        data.kspace.configuration,
        data.kspace.inner_potential,
        **data.kspace.angle_params,
    )
    return alpha, hv


def _exact_cut_other_coord(
    data: xarray.DataArray,
    alpha_target: xarray.DataArray,
    *,
    hv_target: xarray.DataArray | None = None,
) -> xarray.DataArray:
    if hv_target is None:
        beta_target = data.beta
        kinetic_target = data.kspace._kinetic_energy
    else:
        hv_dim = str(data.hv.dims[0])
        if hv_dim in data.beta.dims:
            beta_target = data.beta.interp({hv_dim: hv_target})
        else:
            beta_target = data.beta.broadcast_like(hv_target)
        kinetic_target = (
            hv_target - data.kspace.work_function + data.kspace._binding_energy
        )

    return erlab.analysis.kspace._exact_other_axis_momentum(
        alpha_target,
        beta_target,
        kinetic_target,
        data.kspace.configuration,
        **data.kspace.angle_params,
    )


def _exact_hvdep_other_coord(
    data: xarray.DataArray, slit_momentum: np.ndarray, kz: np.ndarray
) -> xarray.DataArray:
    slit_axis = data.kspace.slit_axis
    slit_value = xarray.DataArray(
        slit_momentum, dims=slit_axis, coords={slit_axis: slit_momentum}
    )
    kz_value = xarray.DataArray(kz, dims="kz", coords={"kz": kz})
    return erlab.analysis.kspace.exact_hv_cut_coords(
        slit_value,
        kz_value,
        data.beta,
        data.hv,
        data.kspace._kinetic_energy,
        data.alpha,
        data.kspace.configuration,
        data.kspace.inner_potential,
        **data.kspace.angle_params,
    )[2]


def _overlay_subset(data: xarray.DataArray) -> xarray.DataArray:
    slit_axis = data.kspace.slit_axis
    slit_idx = np.unique(np.linspace(0, data.sizes[slit_axis] - 1, 4, dtype=int))
    eV_idx = np.unique(np.linspace(0, data.sizes["eV"] - 1, 3, dtype=int))
    return data.isel({slit_axis: slit_idx, "eV": eV_idx})


def _legacy_hv_to_kz(data: xarray.DataArray, hv_values) -> xarray.DataArray:
    hv = xarray.DataArray(np.asarray(hv_values), dims="hv", coords={"hv": hv_values})
    kinetic = hv - data.kspace.work_function + data.eV
    ang2k, k2ang = erlab.analysis.kspace.get_kconv_func(
        kinetic, data.kspace.configuration, data.kspace.angle_params
    )
    kx, ky = ang2k(*k2ang(data.kx, data.ky))
    return erlab.analysis.kspace.kz_func(kinetic, data.kspace.inner_potential, kx, ky)


def _self_consistent_hv_to_kz(data: xarray.DataArray, hv_values) -> xarray.DataArray:
    hv = xarray.DataArray(np.asarray(hv_values), dims="hv", coords={"hv": hv_values})
    kinetic = hv - data.kspace.work_function + data.eV
    slit_coord = data[data.kspace.slit_axis]
    other_coord = data.coords[data.kspace.other_axis]
    target_order = ("hv", "eV", data.kspace.slit_axis)

    if "kz" not in other_coord.dims:
        if data.kspace.slit_axis == "kx":
            return erlab.analysis.kspace.kz_func(
                kinetic, data.kspace.inner_potential, slit_coord, other_coord
            ).transpose(*target_order)
        return erlab.analysis.kspace.kz_func(
            kinetic, data.kspace.inner_potential, other_coord, slit_coord
        ).transpose(*target_order)

    root = xarray.apply_ufunc(
        functools.partial(
            erlab.accessors.kspace._solve_hv_to_kz_roots_2d,
            inner_potential=data.kspace.inner_potential,
            slit_axis=data.kspace.slit_axis,
        ),
        data.kz,
        other_coord,
        kinetic,
        slit_coord,
        input_core_dims=[
            ("kz",),
            (data.kspace.slit_axis, "kz"),
            (),
            (data.kspace.slit_axis,),
        ],
        output_core_dims=[(data.kspace.slit_axis,)],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {data.kspace.slit_axis: slit_coord.size}},
        output_dtypes=[np.float64],
    )
    return root.transpose(*target_order)


def _legacy_hvdep_targets(
    data: xarray.DataArray, slit_momentum: np.ndarray, kz: np.ndarray
) -> tuple[xarray.DataArray, xarray.DataArray]:
    slit_axis = data.kspace.slit_axis
    other_axis = data.kspace.other_axis
    other_lims = data.kspace.estimate_bounds()[other_axis]
    other_mean = np.array([(other_lims[0] + other_lims[1]) / 2.0])

    slit_value = xarray.DataArray(
        slit_momentum, dims=slit_axis, coords={slit_axis: slit_momentum}
    )
    other_value = xarray.DataArray(
        other_mean, dims=other_axis, coords={other_axis: other_mean}
    )
    kz_value = xarray.DataArray(kz, dims="kz", coords={"kz": kz})
    kperp = erlab.analysis.kspace.kperp_from_kz(kz_value, data.kspace.inner_potential)

    if slit_axis == "kx":
        alpha, _ = data.kspace._inverse_func(slit_value, other_value, kperp)
        hv = erlab.analysis.kspace.hv_func(
            slit_value,
            other_value,
            kz_value,
            data.kspace.inner_potential,
            data.kspace.work_function,
            data.kspace._binding_energy,
        )
    else:
        alpha, _ = data.kspace._inverse_func(other_value, slit_value, kperp)
        hv = erlab.analysis.kspace.hv_func(
            other_value,
            slit_value,
            kz_value,
            data.kspace.inner_potential,
            data.kspace.work_function,
            data.kspace._binding_energy,
        )

    hv = hv.squeeze(other_axis, drop=True)
    alpha = alpha.squeeze(other_axis, drop=True).broadcast_like(hv)
    return alpha, hv


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


def _assert_kconv_case(
    request: pytest.FixtureRequest,
    *,
    use_dask: bool,
    energy_axis: str,
    data_type: str,
    configuration: AxesConfiguration,
    extra_dims: int,
    expect_error: bool = False,
) -> None:
    data = request.getfixturevalue(data_type).copy(deep=True)

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

    if expect_error:
        with pytest.raises(
            ValueError,
            match=r"Energy axis of photon energy dependent data must be in "
            r"binding energy.",
        ):
            data.kspace.convert(silent=False)
        return

    expected = request.getfixturevalue(
        f"config_{configuration.value}_{data_type.replace('angle', 'k')}"
    )

    if energy_axis == "kinetic":
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
            assert set(kconv.shape) == set(expected.shape)
        else:
            assert set(kconv.shape) == {*expected.shape, 2}

    assert isinstance(kconv, xarray.DataArray)
    assert not kconv.isnull().all()


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no-dask"])
@pytest.mark.parametrize("energy_axis", ["kinetic", "binding"])
@pytest.mark.parametrize("data_type", ["anglemap", "cut"])
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
    request: pytest.FixtureRequest,
) -> None:
    _assert_kconv_case(
        request,
        use_dask=use_dask,
        energy_axis=energy_axis,
        data_type=data_type,
        configuration=configuration,
        extra_dims=extra_dims,
    )


@pytest.mark.parametrize(
    ("configuration", "use_dask", "extra_dims"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            False,
            0,
            id="Type1-binding-no-dask-extra0",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            True,
            1,
            id="Type2DA-binding-dask-extra1",
        ),
    ],
)
def test_kconv_hvdep_binding_smoke(
    configuration: AxesConfiguration,
    use_dask: bool,
    extra_dims: int,
    request: pytest.FixtureRequest,
) -> None:
    _assert_kconv_case(
        request,
        use_dask=use_dask,
        energy_axis="binding",
        data_type="hvdep",
        configuration=configuration,
        extra_dims=extra_dims,
    )


@pytest.mark.parametrize(
    ("configuration", "use_dask"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            False,
            id="Type1-kinetic-no-dask",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            True,
            id="Type2DA-kinetic-dask",
        ),
    ],
)
def test_kconv_hvdep_kinetic_rejects(
    configuration: AxesConfiguration,
    use_dask: bool,
    request: pytest.FixtureRequest,
) -> None:
    _assert_kconv_case(
        request,
        use_dask=use_dask,
        energy_axis="kinetic",
        data_type="hvdep",
        configuration=configuration,
        extra_dims=0,
        expect_error=True,
    )


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


def test_missing_kspace_parameter_accessors_warn(anglemap) -> None:
    data = anglemap.copy(deep=True)
    data.attrs.pop("sample_workfunction", None)
    data.attrs.pop("inner_potential", None)

    with pytest.warns(
        UserWarning,
        match=r"Work function not found in data attributes, assuming 4\.5 eV",
    ):
        assert data.kspace.work_function == pytest.approx(4.5)

    with pytest.warns(
        UserWarning,
        match=r"Inner potential not found in data attributes, assuming 10 eV",
    ):
        assert data.kspace.inner_potential == pytest.approx(10.0)


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


@pytest.mark.parametrize(
    ("configuration", "xi", "offsets"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            7.25,
            {"delta": 12.5, "xi": 2.5, "beta": -3.75},
            id="type1",
        ),
        pytest.param(
            AxesConfiguration.Type2,
            -6.5,
            {"delta": -8.0, "xi": -1.75, "beta": 2.25},
            id="type2",
        ),
    ],
)
def test_convert_cut_uses_exact_slit_inverse_for_alpha_field(
    cut,
    configuration: AxesConfiguration,
    xi: float,
    offsets: dict[str, float],
) -> None:
    data = _make_alpha_field_cut(cut, configuration, xi=xi, offsets=offsets)

    converted = data.kspace.convert(silent=True)
    slit_axis = data.kspace.slit_axis

    expected_alpha = _exact_cut_target_alpha(data, converted[slit_axis].values)
    expected = _mask_alpha_domain(data, expected_alpha).transpose(*converted.dims)
    legacy = _mask_alpha_domain(
        data, _legacy_cut_target_alpha(data, converted[slit_axis].values)
    ).transpose(*converted.dims)

    finite = np.isfinite(converted.values)
    assert finite.any()
    assert np.allclose(converted.values[finite], expected.values[finite])
    assert not np.allclose(
        converted.values[finite],
        legacy.values[finite],
        equal_nan=True,
    )
    xarray.testing.assert_allclose(
        converted.coords[data.kspace.other_axis].reset_coords(drop=True),
        _exact_cut_other_coord(data, expected_alpha)
        .transpose(*converted.coords[data.kspace.other_axis].dims)
        .reset_coords(drop=True),
    )


def test_convert_cut_exact_slit_inverse_raises_when_multivalued(cut) -> None:
    data = _make_alpha_field_cut(
        cut,
        AxesConfiguration.Type1,
        xi=90.0,
        offsets={"delta": 0.0, "xi": 0.0, "beta": 0.0},
        beta=0.0,
    )

    with pytest.raises(ValueError, match="non-physical angle value or offset"):
        data.kspace.convert(silent=True)


def test_convert_cut_exact_slit_inverse_returns_nan_out_of_domain(cut) -> None:
    data = _make_alpha_field_cut(
        cut,
        AxesConfiguration.Type1,
        xi=7.25,
        offsets={"delta": 12.5, "xi": 2.5, "beta": -3.75},
    )

    lims = data.kspace.estimate_bounds()["kx"]
    out = data.kspace.convert(
        silent=True, kx=np.linspace(lims[0] - 0.1, lims[1] + 0.1, 64)
    )

    assert out.isel(kx=0).isnull().all()
    assert out.isel(kx=-1).isnull().all()
    assert not out.isel(kx=slice(1, -1)).isnull().all()


def test_convert_cut_with_singleton_beta_dim_uses_exact_path(cut) -> None:
    data = _make_alpha_field_cut(
        cut,
        AxesConfiguration.Type1,
        xi=7.25,
        offsets={"delta": 12.5, "xi": 2.5, "beta": -3.75},
    )
    with_beta_dim = data.expand_dims(beta=[float(data.beta)])

    converted = with_beta_dim.kspace.convert(silent=True)
    reference = data.kspace.convert(silent=True)

    assert "beta" not in converted.dims
    xarray.testing.assert_allclose(converted, reference)


def test_convert_cut_with_zero_other_axis_collapses_coord_to_scalar(cut) -> None:
    data = _make_alpha_field_cut(
        cut,
        AxesConfiguration.Type1,
        xi=0.0,
        offsets={"delta": 0.0, "xi": 0.0, "beta": 0.0},
        beta=0.0,
    )

    converted = data.kspace.convert(silent=True)
    other_coord = converted.coords[data.kspace.other_axis]

    assert other_coord.ndim == 0
    assert np.isclose(float(other_coord), 0.0)


@pytest.mark.parametrize(
    ("configuration", "xi", "chi", "offsets"),
    [
        pytest.param(
            AxesConfiguration.Type1DA,
            5.25,
            6.5,
            {"delta": 9.0, "chi": 2.0, "xi": 1.75},
            id="type1da",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            6.75,
            -4.0,
            {"delta": -5.0, "chi": 1.25, "xi": 0.75},
            id="type2da",
        ),
    ],
)
def test_convert_cut_uses_exact_da_inverse_for_alpha_field(
    cut,
    configuration: AxesConfiguration,
    xi: float,
    chi: float,
    offsets: dict[str, float],
) -> None:
    data = _make_alpha_field_cut(cut, configuration, xi=xi, chi=chi, offsets=offsets)

    converted = data.kspace.convert(silent=True)
    slit_axis = data.kspace.slit_axis

    expected_alpha = _mask_alpha_domain(
        data, _exact_cut_target_alpha(data, converted[slit_axis].values)
    ).transpose(*converted.dims)

    finite = np.isfinite(converted.values)
    assert finite.any()
    assert np.allclose(converted.values[finite], expected_alpha.values[finite])
    xarray.testing.assert_allclose(
        converted.coords[data.kspace.other_axis].reset_coords(drop=True),
        _exact_cut_other_coord(data, expected_alpha).reset_coords(drop=True),
    )


def test_convert_cut_exact_da_inverse_raises_when_noninvertible(
    cut, monkeypatch
) -> None:
    data = _make_alpha_field_cut(
        cut,
        AxesConfiguration.Type1DA,
        xi=5.25,
        chi=6.5,
        offsets={"delta": 9.0, "chi": 2.0, "xi": 1.75},
    )

    def _raise(*args, **kwargs):
        raise ValueError(
            "Exact cut conversion is multi-valued over the measured interval."
        )

    monkeypatch.setattr(erlab.analysis.kspace, "exact_cut_alpha", _raise)

    with pytest.raises(ValueError, match="multi-valued"):
        data.kspace.convert(silent=True)


@pytest.mark.parametrize(
    ("configuration", "xi", "offsets"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            0.0,
            {"delta": 0.0, "xi": 0.0, "beta": 0.0},
            id="type1",
        ),
        pytest.param(
            AxesConfiguration.Type2,
            0.0,
            {"delta": 0.0, "xi": 0.0, "beta": 0.0},
            id="type2",
        ),
    ],
)
def test_convert_hv_cut_uses_exact_slit_inverse_for_alpha_field(
    hvdep,
    configuration: AxesConfiguration,
    xi: float,
    offsets: dict[str, float],
) -> None:
    data = _make_field_hvdep_cut(
        hvdep, configuration, field="alpha", xi=xi, offsets=offsets
    )

    converted = data.kspace.convert(silent=True)
    slit_axis = data.kspace.slit_axis

    expected_alpha, legacy_alpha = (
        _exact_hvdep_targets(data, converted[slit_axis].values, converted.kz.values)[0],
        _legacy_hvdep_targets(data, converted[slit_axis].values, converted.kz.values)[
            0
        ],
    )

    expected = expected_alpha.transpose(*converted.dims)
    legacy = legacy_alpha.transpose(*converted.dims)

    finite = np.isfinite(converted.values)
    assert finite.any()
    assert np.allclose(converted.values[finite], expected.values[finite])
    assert not np.allclose(
        converted.values[finite],
        legacy.values[finite],
        equal_nan=True,
    )
    xarray.testing.assert_allclose(
        converted.coords[data.kspace.other_axis].reset_coords(drop=True),
        _exact_hvdep_other_coord(
            data, converted[slit_axis].values, converted.kz.values
        ).reset_coords(drop=True),
    )


@pytest.mark.parametrize(
    ("configuration", "xi", "offsets"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            0.0,
            {"delta": 0.0, "xi": 0.0, "beta": 0.0},
            id="type1",
        ),
        pytest.param(
            AxesConfiguration.Type2,
            0.0,
            {"delta": 0.0, "xi": 0.0, "beta": 0.0},
            id="type2",
        ),
    ],
)
def test_convert_hv_cut_uses_exact_slit_inverse_for_hv_field(
    hvdep,
    configuration: AxesConfiguration,
    xi: float,
    offsets: dict[str, float],
) -> None:
    data = _make_field_hvdep_cut(
        hvdep, configuration, field="hv", xi=xi, offsets=offsets
    )

    converted = data.kspace.convert(silent=True)
    slit_axis = data.kspace.slit_axis

    expected_hv, legacy_hv = (
        _exact_hvdep_targets(data, converted[slit_axis].values, converted.kz.values)[1],
        _legacy_hvdep_targets(data, converted[slit_axis].values, converted.kz.values)[
            1
        ],
    )

    expected = expected_hv.transpose(*converted.dims)
    legacy = legacy_hv.transpose(*converted.dims)

    finite = np.isfinite(converted.values)
    assert finite.any()
    assert np.allclose(converted.values[finite], expected.values[finite])
    assert not np.allclose(
        converted.values[finite],
        legacy.values[finite],
        equal_nan=True,
    )


def test_convert_hv_cut_exact_slit_inverse_raises_when_exact_hv_path_fails(
    hvdep, monkeypatch
) -> None:
    data = _make_field_hvdep_cut(
        hvdep,
        AxesConfiguration.Type1,
        field="alpha",
        xi=0.0,
        offsets={"delta": 0.0, "xi": 0.0, "beta": 0.0},
    )

    def _raise(*args, **kwargs):
        raise ValueError(
            "Exact hv-cut conversion is multi-valued over the measured hv range."
        )

    monkeypatch.setattr(erlab.analysis.kspace, "exact_hv_cut_coords", _raise)

    with pytest.raises(ValueError, match="multi-valued"):
        data.kspace.convert(silent=True)


def test_convert_hv_cut_exact_slit_inverse_returns_nan_out_of_domain(hvdep) -> None:
    data = _make_field_hvdep_cut(
        hvdep,
        AxesConfiguration.Type1,
        field="hv",
        xi=0.0,
        offsets={"delta": 0.0, "xi": 0.0, "beta": 0.0},
    )

    lims = data.kspace.estimate_bounds()["kz"]
    out = data.kspace.convert(
        silent=True, kz=np.linspace(lims[0] - 0.2, lims[1] + 0.2, 64)
    )

    assert out.isel(kz=0).isnull().all()
    assert out.isel(kz=-1).isnull().all()
    assert not out.isel(kz=slice(1, -1)).isnull().all()


@pytest.mark.parametrize(
    ("configuration", "xi", "chi", "offsets"),
    [
        pytest.param(
            AxesConfiguration.Type1DA,
            5.25,
            6.5,
            {"delta": 9.0, "chi": 2.0, "xi": 1.75},
            id="type1da",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            6.75,
            -4.0,
            {"delta": -5.0, "chi": 1.25, "xi": 0.75},
            id="type2da",
        ),
    ],
)
def test_convert_hv_cut_uses_exact_da_inverse_for_alpha_field(
    hvdep,
    configuration: AxesConfiguration,
    xi: float,
    chi: float,
    offsets: dict[str, float],
) -> None:
    data = _make_field_hvdep_cut(
        hvdep, configuration, field="alpha", xi=xi, chi=chi, offsets=offsets
    )

    converted = data.kspace.convert(silent=True)
    slit_axis = data.kspace.slit_axis
    expected_alpha, _ = _exact_hvdep_targets(
        data, converted[slit_axis].values, converted.kz.values
    )
    expected = expected_alpha.transpose(*converted.dims)

    finite = np.isfinite(converted.values)
    assert finite.any()
    assert np.allclose(converted.values[finite], expected.values[finite])
    xarray.testing.assert_allclose(
        converted.coords[data.kspace.other_axis].reset_coords(drop=True),
        _exact_hvdep_other_coord(
            data, converted[slit_axis].values, converted.kz.values
        ).reset_coords(drop=True),
    )


@pytest.mark.parametrize(
    ("configuration", "xi", "chi", "offsets"),
    [
        pytest.param(
            AxesConfiguration.Type1DA,
            5.25,
            6.5,
            {"delta": 9.0, "chi": 2.0, "xi": 1.75},
            id="type1da",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            6.75,
            -4.0,
            {"delta": -5.0, "chi": 1.25, "xi": 0.75},
            id="type2da",
        ),
    ],
)
def test_convert_hv_cut_uses_exact_da_inverse_for_hv_field(
    hvdep,
    configuration: AxesConfiguration,
    xi: float,
    chi: float,
    offsets: dict[str, float],
) -> None:
    data = _make_field_hvdep_cut(
        hvdep, configuration, field="hv", xi=xi, chi=chi, offsets=offsets
    )

    converted = data.kspace.convert(silent=True)
    slit_axis = data.kspace.slit_axis
    expected_hv = _exact_hvdep_targets(
        data, converted[slit_axis].values, converted.kz.values
    )[1].transpose(*converted.dims)

    finite = np.isfinite(converted.values)
    assert finite.any()
    assert np.allclose(converted.values[finite], expected_hv.values[finite])


def test_convert_hv_cut_exact_da_inverse_raises_when_noninvertible(
    hvdep, monkeypatch
) -> None:
    data = _make_field_hvdep_cut(
        hvdep,
        AxesConfiguration.Type1DA,
        field="alpha",
        xi=5.25,
        chi=6.5,
        offsets={"delta": 9.0, "chi": 2.0, "xi": 1.75},
    )

    def _raise(*args, **kwargs):
        raise ValueError(
            "Exact cut conversion is multi-valued over the measured interval."
        )

    monkeypatch.setattr(erlab.analysis.kspace, "exact_hv_cut_coords", _raise)

    with pytest.raises(ValueError, match="multi-valued"):
        data.kspace.convert(silent=True)


@pytest.mark.parametrize(
    ("configuration", "xi", "chi", "offsets"),
    [
        pytest.param(
            AxesConfiguration.Type1,
            0.0,
            None,
            {"delta": 0.0, "xi": 0.0, "beta": 0.0},
            id="type1",
        ),
        pytest.param(
            AxesConfiguration.Type2DA,
            6.75,
            -4.0,
            {"delta": -5.0, "chi": 1.25, "xi": 0.75},
            id="type2da",
        ),
    ],
)
def test_hv_to_kz_exact_off_center_cut_returns_overlay_curve(
    hvdep,
    configuration: AxesConfiguration,
    xi: float,
    chi: float | None,
    offsets: dict[str, float],
) -> None:
    data = _make_field_hvdep_cut(
        hvdep, configuration, field="alpha", xi=xi, chi=chi, offsets=offsets
    )
    converted = _overlay_subset(data.kspace.convert(silent=True))
    hv_values = [30.0, 45.0, 60.0]

    assert "kz" in converted.coords[converted.kspace.other_axis].dims

    out = converted.kspace.hv_to_kz(hv_values)
    expected = _self_consistent_hv_to_kz(converted, hv_values)

    assert "kz" not in out.dims
    xarray.testing.assert_allclose(out, expected)


def test_hv_to_kz_direct_formula_for_scalar_other_axis(cut) -> None:
    data = _make_alpha_field_cut(
        cut,
        AxesConfiguration.Type1,
        xi=0.0,
        offsets={"delta": 0.0, "xi": 0.0, "beta": 0.0},
        beta=0.0,
    )
    data.kspace.inner_potential = 10.0
    converted = data.kspace.convert(silent=True)
    hv_values = [25.0, 35.0, 45.0]

    out = converted.kspace.hv_to_kz(hv_values)
    expected = _self_consistent_hv_to_kz(converted, hv_values)

    assert "kz" not in out.dims
    assert converted.coords[data.kspace.other_axis].ndim == 0
    xarray.testing.assert_allclose(out, expected)


def test_hv_to_kz_falls_back_to_legacy_when_stored_coord_path_is_unavailable(
    hvdep, monkeypatch
) -> None:
    data = _make_field_hvdep_cut(
        hvdep,
        AxesConfiguration.Type1,
        field="alpha",
        xi=0.0,
        offsets={"delta": 0.0, "xi": 0.0, "beta": 0.0},
    )
    converted = _overlay_subset(data.kspace.convert(silent=True))
    hv_values = [30.0, 45.0, 60.0]

    monkeypatch.setattr(
        erlab.accessors.kspace.MomentumAccessor,
        "_hv_to_kz_from_stored_coords",
        lambda self, kinetic: None,
    )

    out = converted.kspace.hv_to_kz(hv_values)
    expected = _legacy_hv_to_kz(converted, hv_values)

    xarray.testing.assert_allclose(out, expected)


def test_hv_to_kz_accepts_iterable_hv(config_1_hvdep) -> None:
    converted = _overlay_subset(config_1_hvdep)
    out = converted.kspace.hv_to_kz([30.0, 45.0, 60.0])
    expected = _self_consistent_hv_to_kz(converted, [30.0, 45.0, 60.0])

    assert out.dims == ("hv", "eV", converted.kspace.slit_axis)
    assert out.hv.size == 3
    xarray.testing.assert_allclose(out, expected)


def test_hv_to_kz_root_candidates_1d_rejects_nonfinite_inputs() -> None:
    out = erlab.accessors.kspace._hv_to_kz_root_candidates_1d(
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
        np.nan,
        0.0,
        inner_potential=10.0,
        slit_axis="kx",
    )

    assert out.size == 0


def test_hv_to_kz_root_candidates_1d_groups_exact_segments(monkeypatch) -> None:
    def _mock_kz_func(kinetic, inner_potential, kx, ky):
        return np.array([0.0, 1.0, 6.0, 3.0, 8.0], dtype=float)

    monkeypatch.setattr(erlab.analysis.kspace, "kz_func", _mock_kz_func)

    out = erlab.accessors.kspace._hv_to_kz_root_candidates_1d(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.zeros(5),
        25.0,
        0.0,
        inner_potential=10.0,
        slit_axis="kx",
    )

    assert np.allclose(out, [0.5, 3.0])


def test_solve_hv_to_kz_roots_2d_returns_nan_without_unique_seed(monkeypatch) -> None:
    lookup = {
        0.0: np.array([], dtype=float),
        1.0: np.array([0.25, 1.25], dtype=float),
        2.0: np.array([0.5, 1.5], dtype=float),
    }

    def _mock_candidates(
        kz_grid,
        other_momentum,
        kinetic_energy,
        slit_momentum,
        *,
        inner_potential: float,
        slit_axis: str,
    ) -> np.ndarray:
        return lookup[float(slit_momentum)]

    monkeypatch.setattr(
        erlab.accessors.kspace, "_hv_to_kz_root_candidates_1d", _mock_candidates
    )

    out = erlab.accessors.kspace._solve_hv_to_kz_roots_2d(
        np.array([0.0]),
        np.zeros((3, 1)),
        30.0,
        np.array([0.0, 1.0, 2.0]),
        inner_potential=10.0,
        slit_axis="kx",
    )

    assert np.isnan(out).all()


def test_solve_hv_to_kz_roots_2d_propagates_closest_branch(monkeypatch) -> None:
    lookup = {
        0.0: np.array([0.2, 2.2], dtype=float),
        1.0: np.array([1.0], dtype=float),
        2.0: np.array([0.9, 3.5], dtype=float),
        3.0: np.array([], dtype=float),
    }

    def _mock_candidates(
        kz_grid,
        other_momentum,
        kinetic_energy,
        slit_momentum,
        *,
        inner_potential: float,
        slit_axis: str,
    ) -> np.ndarray:
        return lookup[float(slit_momentum)]

    monkeypatch.setattr(
        erlab.accessors.kspace, "_hv_to_kz_root_candidates_1d", _mock_candidates
    )

    out = erlab.accessors.kspace._solve_hv_to_kz_roots_2d(
        np.array([0.0]),
        np.zeros((4, 1)),
        30.0,
        np.array([0.0, 1.0, 2.0, 3.0]),
        inner_potential=10.0,
        slit_axis="kx",
    )

    assert np.allclose(out[:3], [0.2, 1.0, 0.9])
    assert np.isnan(out[3])


def test_inverse_exact_cut_without_energy_axis_omits_ev_output(
    cut, monkeypatch
) -> None:
    data = _make_alpha_field_cut(
        cut,
        AxesConfiguration.Type1,
        xi=0.0,
        offsets={"delta": 0.0, "xi": 0.0, "beta": 0.0},
        beta=0.0,
    ).isel(eV=0, drop=True)

    slit_axis = data.kspace.slit_axis
    alpha = xarray.DataArray([0.0], dims=(slit_axis,), coords={slit_axis: [0.0]})
    other = xarray.DataArray(0.0)

    monkeypatch.setattr(
        erlab.accessors.kspace.MomentumAccessor,
        "_kinetic_energy",
        property(lambda self: xarray.DataArray(20.0)),
    )
    monkeypatch.setattr(
        erlab.analysis.kspace,
        "exact_cut_alpha",
        lambda *args, **kwargs: alpha,
    )
    monkeypatch.setattr(
        erlab.analysis.kspace,
        "_exact_other_axis_momentum",
        lambda *args, **kwargs: other,
    )
    monkeypatch.setattr(
        erlab.accessors.kspace.MomentumAccessor,
        "_broadcast_exact_targets",
        lambda self, out_dict, other_momentum: out_dict,
    )

    out = data.kspace._inverse_exact_cut(np.array([0.0]))

    assert "eV" not in out
    assert set(out) == {"alpha"}


def test_hv_to_kz_from_stored_coords_returns_none_without_other_axis(
    config_1_hvdep,
) -> None:
    converted = _overlay_subset(config_1_hvdep)
    converted = converted.drop_vars(converted.kspace.other_axis)
    kinetic = (
        xarray.DataArray([30.0], dims=("hv",), coords={"hv": [30.0]})
        - converted.kspace.work_function
        + converted.eV
    )

    out = converted.kspace._hv_to_kz_from_stored_coords(kinetic)

    assert out is None


def test_hv_to_kz_raises_without_stored_coord_or_legacy_metadata(
    config_1_hvdep, monkeypatch
) -> None:
    converted = _overlay_subset(config_1_hvdep)
    converted = converted.drop_vars(converted.kspace.other_axis)

    def _raise(self, kinetic):
        raise KeyError("missing-coordinate")

    monkeypatch.setattr(
        erlab.accessors.kspace.MomentumAccessor,
        "_hv_to_kz_legacy",
        _raise,
    )

    with pytest.raises(
        ValueError,
        match="requires the orthogonal in-plane momentum coordinate or enough metadata",
    ):
        converted.kspace.hv_to_kz([30.0])
