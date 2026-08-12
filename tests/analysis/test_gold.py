import inspect
import typing

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import erlab
import erlab.analysis.gold as gold_mod
from erlab.analysis.gold import correct_with_edge, edge, poly, quick_fit, spline


def test_spline_forwards_along_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a dummy gold array with dims ('beta', 'eV')
    beta = np.linspace(0.0, 10.0, 11)
    eV = np.linspace(-0.2, 0.2, 51)
    gold = xr.DataArray(
        np.ones((beta.size, eV.size), dtype=float),
        dims=("beta", "eV"),
        coords={"beta": beta, "eV": eV},
    )

    # Stub edge() to return center and stderr along 'beta'
    center_vals = np.linspace(0.0, 0.1, beta.size)
    center_arr = xr.DataArray(center_vals, dims=["beta"], coords={"beta": beta})
    center_stderr = xr.DataArray(
        np.full(beta.shape, 0.01), dims=["beta"], coords={"beta": beta}
    )

    def _stub_edge(
        *_args: typing.Any, **_kwargs: typing.Any
    ) -> tuple[xr.DataArray, xr.DataArray]:
        return center_arr, center_stderr

    monkeypatch.setattr(gold_mod, "edge", _stub_edge)

    # Execute
    result = gold_mod.spline(
        gold,
        along="beta",
        angle_range=(beta.min(), beta.max()),
        eV_range=(eV.min(), eV.max()),
        plot=False,
    )

    # Validate type without importing scipy at module top to avoid hard dependency
    from scipy.interpolate import BSpline  # local import for optional dependency

    assert isinstance(result, BSpline)


def test_range_slice_for_coord_single_point_sorts_bounds() -> None:
    coord = xr.DataArray([0.0], dims=("x",), coords={"x": [0.0]}, name="x")

    out = gold_mod._range_slice_for_coord(coord, (1.0, -1.0))

    assert out == slice(-1.0, 1.0)


@pytest.mark.parametrize(
    ("coord_values", "value_range", "expected"),
    [
        ([-2.0, -1.0, 0.0, 1.0, 2.0], (None, 0.0), [-2.0, -1.0, 0.0]),
        ([-2.0, -1.0, 0.0, 1.0, 2.0], (0.0, None), [0.0, 1.0, 2.0]),
        ([2.0, 1.0, 0.0, -1.0, -2.0], (None, 0.0), [0.0, -1.0, -2.0]),
        ([2.0, 1.0, 0.0, -1.0, -2.0], (0.0, None), [2.0, 1.0, 0.0]),
    ],
)
def test_range_slice_for_coord_supports_open_bounds(
    coord_values: list[float],
    value_range: tuple[float | None, float | None],
    expected: list[float],
) -> None:
    coord = xr.DataArray(
        coord_values, dims=("x",), coords={"x": coord_values}, name="x"
    )

    out = coord.sel(x=gold_mod._range_slice_for_coord(coord, value_range))

    assert out.x.values.tolist() == expected


def test_range_slice_for_coord_nonmonotonic_raises() -> None:
    coord = xr.DataArray(
        [0.0, 2.0, 1.0], dims=("x",), coords={"x": [0.0, 2.0, 1.0]}, name="x"
    )

    with pytest.raises(ValueError, match=r"Coordinate `x` is not monotonic"):
        _ = gold_mod._range_slice_for_coord(coord, (-1.0, 1.0))


@pytest.mark.parametrize(
    "func",
    [
        gold_mod.guess_edge_fit_range,
        gold_mod.edge,
        gold_mod.poly,
        gold_mod.spline,
        gold_mod.resolution,
    ],
)
def test_gold_fit_api_exposes_use_step_edge(func: typing.Callable) -> None:
    parameters = inspect.signature(func).parameters

    assert "use_step_edge" in parameters
    assert "fast" not in parameters


@pytest.mark.parametrize("fast", [False, True])
def test_guess_edge_fit_range_warns_for_deprecated_fast(
    gold: xr.DataArray, fast: bool
) -> None:
    edc = gold.sel(eV=slice(-0.2, 0.2)).sel(alpha=0.0)

    with pytest.warns(FutureWarning, match="Use `use_step_edge` instead"):
        legacy_bounds = gold_mod.guess_edge_fit_range(edc, temp=100.0, fast=fast)
    current_bounds = gold_mod.guess_edge_fit_range(edc, temp=100.0, use_step_edge=fast)

    assert legacy_bounds == current_bounds


@pytest.mark.parametrize(
    "use_step_edge", [True, False], ids=["use_step_edge", "regular"]
)
def test_guess_edge_fit_range(gold: xr.DataArray, use_step_edge: bool) -> None:
    edc = gold.sel(eV=slice(-0.2, 0.2)).sel(alpha=0.0)

    lower, upper = gold_mod.guess_edge_fit_range(
        edc, temp=100.0, use_step_edge=use_step_edge
    )
    selected = edc.sel(eV=slice(lower, upper))

    assert lower < 0.04 < upper
    assert 0 < selected.sizes["eV"] <= edc.sizes["eV"]


def test_edge_model_params_fix_background_slope() -> None:
    _, params = gold_mod._edge_model_and_params(
        temp=10.0,
        resolution=0.02,
        vary_temp=False,
        bkg_slope=False,
        use_step_edge=True,
    )

    assert params["back1"].value == 0.0
    assert params["back1"].vary is False


@pytest.mark.parametrize(
    ("eV", "message"),
    [
        (np.linspace(-0.1, 0.1, 11), "At least 12 finite points are required"),
        (
            np.array(
                [
                    -0.1,
                    -0.08,
                    -0.06,
                    -0.04,
                    -0.02,
                    0.0,
                    0.0,
                    0.02,
                    0.04,
                    0.06,
                    0.08,
                    0.1,
                ]
            ),
            "Energy coordinates must be unique",
        ),
    ],
)
def test_guess_edge_fit_range_validates_energy_coordinate(
    eV: np.ndarray, message: str
) -> None:
    edc = xr.DataArray(np.ones(eV.size), coords={"eV": eV}, dims="eV")

    with pytest.raises(ValueError, match=message):
        gold_mod.guess_edge_fit_range(edc, temp=10.0, use_step_edge=True)


def test_guess_edge_fit_range_uses_temperature_metadata() -> None:
    eV = np.linspace(-0.2, 0.1, 151)
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = model.func(eV, 0.0, 0.01, 0.1, 0.0, 1.0, 0.0)
    edc = xr.DataArray(
        values,
        coords={"eV": eV},
        dims="eV",
        attrs={"sample_temp": 20.0},
    )

    lower, upper = gold_mod.guess_edge_fit_range(edc, use_step_edge=False)

    assert lower < 0.0 < upper


def test_guess_edge_fit_range_temperature_fallback_and_validation() -> None:
    eV = np.linspace(-0.2, 0.1, 151)
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = model.func(eV, 0.0, 0.01, 0.1, 0.0, 1.0, 0.0)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")

    lower, upper = gold_mod.guess_edge_fit_range(edc, use_step_edge=True)
    assert lower < 0.0 < upper

    with pytest.raises(ValueError, match="Temperature not found"):
        gold_mod.guess_edge_fit_range(edc, use_step_edge=False)


def test_guess_edge_fit_range_requires_one_edc() -> None:
    data = xr.DataArray(
        np.ones((2, 12)),
        coords={"alpha": [0.0, 1.0], "eV": np.linspace(-0.1, 0.1, 12)},
        dims=("alpha", "eV"),
    )

    with pytest.raises(ValueError, match="Expected a 1D DataArray"):
        gold_mod.guess_edge_fit_range(data, temp=10.0, use_step_edge=True)


def test_guess_edge_fit_range_uses_model_guess_for_rank_deficient_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eV = np.linspace(-0.3, 0.15, 301)
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = model.func(eV, 0.0, 0.012, 0.15, 0.08, 1.2, -0.12)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")
    original_lstsq = np.linalg.lstsq
    forced_rank_deficiency = False

    def rank_deficient_once(*args: typing.Any, **kwargs: typing.Any):
        nonlocal forced_rank_deficiency
        result = original_lstsq(*args, **kwargs)
        if not forced_rank_deficiency:
            forced_rank_deficiency = True
            coefficients, residuals, rank, singular_values = result
            return coefficients, residuals, rank - 1, singular_values
        return result

    monkeypatch.setattr(np.linalg, "lstsq", rank_deficient_once)

    lower, upper = gold_mod.guess_edge_fit_range(
        edc,
        temp=0.0,
        resolution=2.355 * 0.012,
        use_step_edge=True,
    )

    assert forced_rank_deficiency
    assert lower < 0.0 < upper


def test_guess_edge_fit_range_finds_edge_outside_initial_tail() -> None:
    eV = np.linspace(-0.4, 0.2, 401)
    center = -0.12
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = model.func(eV, center, 0.006, 0.15, 0.08, 1.2, -0.12)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")

    lower, upper = gold_mod.guess_edge_fit_range(
        edc,
        temp=0.0,
        resolution=2.355 * 0.006,
        use_step_edge=True,
    )

    assert lower < center < upper
    assert lower == pytest.approx(center - 6 * 0.006, abs=2 * (eV[1] - eV[0]))
    assert upper == eV[-1]


def test_guess_edge_fit_range_falls_back_when_seed_fit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eV = np.linspace(-0.3, 0.15, 301)
    model_class = erlab.analysis.fit.models.StepEdgeModel
    values = model_class().func(eV, 0.0, 0.012, 0.15, 0.08, 1.2, -0.12)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")
    kwargs = {"temp": 0.0, "resolution": 2.355 * 0.012, "use_step_edge": True}

    def fail_fit(*_args: typing.Any, **_kwargs: typing.Any) -> typing.NoReturn:
        raise ValueError("synthetic seed fit failure")

    monkeypatch.setattr(model_class, "fit", fail_fit)

    bounds = gold_mod.guess_edge_fit_range(edc, **kwargs)

    assert bounds == pytest.approx((eV[0], eV[-1]))


def test_guess_edge_fit_range_falls_back_for_unsuccessful_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eV = np.linspace(-0.3, 0.15, 301)
    model_class = erlab.analysis.fit.models.StepEdgeModel
    values = model_class().func(eV, 0.0, 0.012, 0.15, 0.08, 1.2, -0.12)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")

    class FailedResult:
        def __init__(self) -> None:
            self.success = False
            self.best_values = {"center": 0.0, "sigma": 0.01}

    monkeypatch.setattr(model_class, "fit", lambda *_args, **_kwargs: FailedResult())

    bounds = gold_mod.guess_edge_fit_range(
        edc,
        temp=0.0,
        resolution=2.355 * 0.012,
        bkg_slope=False,
        use_step_edge=True,
    )

    assert bounds == pytest.approx((eV[0], eV[-1]))


@pytest.mark.parametrize("candidate_index", [1, -2])
def test_guess_edge_fit_range_uses_full_range_without_candidate_support(
    monkeypatch: pytest.MonkeyPatch,
    candidate_index: int,
) -> None:
    eV = np.linspace(-0.1, 0.1, 41)
    values = np.where(eV <= 0.0, 1.0, 0.0)
    monkeypatch.setattr(
        gold_mod,
        "_find_falling_edge_candidates",
        lambda *_args, **_kwargs: [(float(eV[candidate_index]), 0.0)],
    )

    def fail_fit(*_args: typing.Any, **_kwargs: typing.Any) -> typing.NoReturn:
        raise ValueError("synthetic trial fit failure")

    monkeypatch.setattr(erlab.analysis.fit.models.StepEdgeModel, "fit", fail_fit)

    bounds = gold_mod._guess_edge_fit_range(
        eV,
        values,
        temp=0.0,
        resolution=0.01,
        use_step_edge=True,
    )

    assert bounds == pytest.approx((eV[0], eV[-1]))


def test_guess_edge_fit_range_rejects_candidate_without_terminal_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eV = np.linspace(-0.1, 0.1, 41)
    values = np.ones(eV.size)
    monkeypatch.setattr(
        gold_mod,
        "_find_falling_edge_candidates",
        lambda *_args, **_kwargs: [(0.0, 0.0)],
    )

    def fail_fit(*_args: typing.Any, **_kwargs: typing.Any) -> typing.NoReturn:
        raise ValueError("synthetic trial fit failure")

    monkeypatch.setattr(erlab.analysis.fit.models.StepEdgeModel, "fit", fail_fit)

    with pytest.raises(ValueError, match="No falling edge was detected"):
        gold_mod._guess_edge_fit_range(
            eV,
            values,
            temp=0.0,
            resolution=0.01,
            use_step_edge=True,
        )


def test_guess_edge_fit_range_default_requires_finite_energy() -> None:
    with pytest.raises(ValueError, match="Energy coordinates must contain finite"):
        gold_mod._guess_edge_fit_range_or_default(
            np.full(12, np.nan),
            np.ones(12),
            temp=10.0,
            resolution=0.02,
            use_step_edge=True,
        )


def test_guess_edge_fit_range_rejects_rising_edge() -> None:
    eV = np.linspace(-0.3, 0.15, 301)
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = -model.func(eV, 0.0, 0.012, 0.15, 0.08, 1.2, -0.12)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")

    with pytest.raises(ValueError, match="No falling edge was detected"):
        gold_mod.guess_edge_fit_range(
            edc,
            temp=0.0,
            resolution=2.355 * 0.012,
            use_step_edge=True,
        )


def test_guess_edge_fit_range_uses_terminal_edge_after_occupied_peak() -> None:
    eV = np.linspace(-0.4, 0.15, 551)
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = model.func(eV, 0.0, 0.012, 0.1, 0.0, 1.0, 0.0)
    values += 5 * 0.01**2 / ((eV + 0.18) ** 2 + 0.01**2)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")

    lower, upper = gold_mod.guess_edge_fit_range(
        edc,
        temp=0.0,
        resolution=2.355 * 0.01,
        use_step_edge=True,
    )

    assert -0.18 < lower < 0.0 < upper


def test_guess_edge_fit_range_uses_multiscale_terminal_edge() -> None:
    rng = np.random.default_rng(13)
    eV = np.linspace(-0.5, 0.15, 501)
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = model.func(eV, 0.0, 0.013, 0.05, 0.0, 0.12, 0.0)
    values += 0.5 * 0.03**2 / ((eV + 0.24) ** 2 + 0.03**2)
    values += rng.normal(0.0, 0.03, eV.size)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")

    lower, upper = gold_mod.guess_edge_fit_range(
        edc,
        temp=0.0,
        resolution=2.355 * 0.013,
        use_step_edge=True,
    )

    assert -0.15 < lower < 0.0 < upper


def test_guess_edge_fit_range_estimates_width_from_broad_edge() -> None:
    eV = np.linspace(-0.35, 0.2, 551)
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = model.func(eV, 0.0, 0.05, 0.1, 0.0, 1.0, 0.0)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")

    lower, upper = gold_mod.guess_edge_fit_range(
        edc,
        temp=0.0,
        resolution=2.355 * 0.005,
        use_step_edge=True,
    )

    assert lower == pytest.approx(-0.2, abs=2 * (eV[1] - eV[0]))
    assert upper == eV[-1]


def test_guess_edge_fit_range_supports_edge_near_upper_bound() -> None:
    eV = np.linspace(-0.2, 0.035, 96)
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = model.func(eV, 0.0, 0.012, 0.1, 0.0, 1.0, 0.0)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")

    lower, upper = gold_mod.guess_edge_fit_range(
        edc,
        temp=0.0,
        resolution=2.355 * 0.01,
        use_step_edge=True,
    )

    assert lower < 0.0 < upper
    assert upper == eV[-1]


def test_guess_edge_fit_range_uses_full_range_with_short_occupied_side() -> None:
    eV = np.linspace(-0.02, 0.2, 221)
    model = erlab.analysis.fit.models.StepEdgeModel()
    values = model.func(eV, 0.0, 0.006, 0.15, 0.08, 1.2, -0.12)
    edc = xr.DataArray(values, coords={"eV": eV}, dims="eV")

    bounds = gold_mod.guess_edge_fit_range(
        edc,
        temp=0.0,
        resolution=2.355 * 0.006,
        use_step_edge=True,
    )

    assert bounds == pytest.approx((eV[0], eV[-1]))


@pytest.mark.parametrize("use_dask", [False, True], ids=["no_dask", "dask"])
@pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalized"])
def test_edge_adaptive_uses_per_edc_ranges(
    gold: xr.DataArray, use_dask: bool, normalize: bool
) -> None:
    if use_dask:
        gold = gold.chunk(alpha=1, eV=25)

    result = edge(
        gold,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        adaptive=True,
        temp=100.0,
        use_step_edge=True,
        normalize=normalize,
        progress=False,
        return_full=True,
        parallel_kw={"backend": "threading", "n_jobs": 1, "return_as": "list"},
    )
    if use_dask:
        result = result.compute()

    point_counts = result.modelfit_data.count("eV")
    assert np.unique(point_counts).size > 1
    assert point_counts.max() <= gold.sel(eV=slice(-0.2, 0.2)).sizes["eV"]
    assert point_counts.min() < point_counts.max()


@pytest.mark.parametrize("use_dask", [False, True], ids=["no_dask", "dask"])
def test_edge_adaptive_falls_back_for_one_edc(
    gold: xr.DataArray,
    monkeypatch: pytest.MonkeyPatch,
    use_dask: bool,
) -> None:
    original = gold_mod._guess_edge_fit_range
    target_alpha = 0.0
    target_values = np.asarray(gold.sel(alpha=target_alpha).sel(eV=slice(-0.2, 0.2)))

    def fail_target_range(
        x: np.ndarray, y: np.ndarray, **kwargs: typing.Any
    ) -> tuple[float, float]:
        if np.array_equal(y, target_values):
            raise ValueError("synthetic range failure")
        return original(x, y, **kwargs)

    monkeypatch.setattr(gold_mod, "_guess_edge_fit_range", fail_target_range)
    if use_dask:
        gold = gold.chunk(alpha=1)

    result = edge(
        gold,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        adaptive=True,
        temp=100.0,
        use_step_edge=True,
        normalize=False,
        progress=False,
        return_full=True,
        parallel_kw={"backend": "threading", "n_jobs": 1, "return_as": "list"},
    )
    if use_dask:
        result = result.compute()

    point_counts = result.modelfit_data.count("eV")
    full_count = gold.sel(eV=slice(-0.2, 0.2)).sizes["eV"]
    assert point_counts.sel(alpha=target_alpha) == full_count
    assert point_counts.min() < full_count


def test_edge_adaptive_requires_skipna(gold: xr.DataArray) -> None:
    with (
        pytest.warns(FutureWarning, match="Use `use_step_edge` instead"),
        pytest.raises(ValueError, match="adaptive fitting requires skipna=True"),
    ):
        edge(
            gold,
            angle_range=(-15, 15),
            eV_range=(-0.2, 0.2),
            adaptive=True,
            temp=100.0,
            fast=True,
            progress=False,
            skipna=False,
        )


@pytest.mark.parametrize("wrapper", [gold_mod.poly, gold_mod.spline])
def test_edge_model_wrapper_forwards_options(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: typing.Callable[..., typing.Any],
) -> None:
    beta = np.linspace(0.0, 10.0, 11)
    eV = np.linspace(-0.2, 0.2, 51)
    data = xr.DataArray(
        np.ones((beta.size, eV.size)),
        dims=("beta", "eV"),
        coords={"beta": beta, "eV": eV},
    )
    center = xr.DataArray(np.linspace(0.0, 0.1, beta.size), coords={"beta": beta})
    stderr = xr.ones_like(center)
    received: dict[str, typing.Any] = {}

    def stub_edge(*_args: typing.Any, **kwargs: typing.Any):
        received.update(kwargs)
        return center, stderr

    monkeypatch.setattr(gold_mod, "edge", stub_edge)

    with pytest.warns(FutureWarning, match="Use `use_step_edge` instead"):
        wrapper(
            data,
            along="beta",
            angle_range=(0.0, 10.0),
            eV_range=(-0.2, 0.2),
            adaptive=True,
            fast=True,
            plot=False,
        )

    assert received["adaptive"] is True
    assert received["use_step_edge"] is True


@pytest.mark.parametrize(
    "parallel_kw", [None, {"return_as": "list"}], ids=["generator", "list"]
)
@pytest.mark.parametrize(
    "use_step_edge", [True, False], ids=["use_step_edge", "regular"]
)
@pytest.mark.parametrize("use_dask", [False, True], ids=["no_dask", "dask"])
def test_poly(gold, parallel_kw: dict, use_step_edge: bool, use_dask: bool) -> None:
    if use_dask:
        gold = gold.chunk(alpha=1)
    if parallel_kw:
        parallel_kw["backend"] = "threading"
    else:
        parallel_kw = {"backend": "threading"}
    res = poly(
        gold,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        temp=100.0,
        use_step_edge=use_step_edge,
        vary_temp=False,
        degree=2,
        plot=True,
        parallel_kw=parallel_kw,
    )
    plt.close()

    assert_allclose(
        np.array(list(res.modelfit_results.item().best_values.values())),
        np.array([0.04, 1e-5, -3e-4]),
        atol=1e-2,
    )

    corr_shift = correct_with_edge(gold, res, shift_coords=True, plot=False)
    assert_allclose(
        corr_shift.eV[[0, -1]], np.array([-1.34295302, 0.33221477]), atol=1e-5
    )

    corr_noshift = correct_with_edge(gold, res, shift_coords=False, plot=False)
    assert_allclose(corr_noshift.eV, gold.eV)

    res = res.drop_vars("modelfit_results")

    xr.testing.assert_allclose(
        corr_shift, correct_with_edge(gold, res, shift_coords=True, plot=False)
    )
    xr.testing.assert_allclose(
        corr_noshift, correct_with_edge(gold, res, shift_coords=False, plot=False)
    )


@pytest.mark.parametrize(
    "parallel_kw", [None, {"return_as": "list"}], ids=["generator", "list"]
)
@pytest.mark.parametrize(
    "use_step_edge", [True, False], ids=["use_step_edge", "regular"]
)
@pytest.mark.parametrize("use_dask", [False, True], ids=["no_dask", "dask"])
def test_poly_nd(gold, parallel_kw: dict, use_step_edge: bool, use_dask: bool) -> None:
    gold_nd = gold.expand_dims(
        {"beta": np.array([-1.0, 0.0, 1.0]), "hv": np.array([20.0, 21.0])}
    )
    if use_dask:
        gold_nd = gold_nd.chunk({"beta": 1, "hv": 1})
    if parallel_kw:
        parallel_kw["backend"] = "threading"
    else:
        parallel_kw = {"backend": "threading"}
    res = poly(
        gold_nd,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        temp=100.0,
        use_step_edge=use_step_edge,
        vary_temp=False,
        degree=2,
        parallel_kw=parallel_kw,
        plot=False,
    )
    assert res.beta.size == 3
    assert res.hv.size == 2

    corr_shift = correct_with_edge(gold_nd, res, shift_coords=True, plot=False)
    assert_allclose(
        corr_shift.eV[[0, -1]], np.array([-1.34295302, 0.33221477]), atol=1e-5
    )

    corr_noshift = correct_with_edge(gold_nd, res, shift_coords=False, plot=False)
    assert_allclose(corr_noshift.eV, gold_nd.eV)

    res = res.drop_vars("modelfit_results")

    xr.testing.assert_allclose(
        corr_shift, correct_with_edge(gold_nd, res, shift_coords=True, plot=False)
    )
    xr.testing.assert_allclose(
        corr_noshift, correct_with_edge(gold_nd, res, shift_coords=False, plot=False)
    )


def test_spline(gold) -> None:
    spl = spline(
        gold,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        temp=100.0,
        vary_temp=False,
        use_step_edge=True,
        lam=None,
        plot=True,
        parallel_kw={"backend": "threading"},
    )
    plt.close()

    assert_allclose(spl(0.0), 0.04, atol=1e-4)

    correct_with_edge(gold, spl, shift_coords=True, plot=False)
    correct_with_edge(gold, spl, shift_coords=False, plot=True)
    plt.close()


def test_poly_crop_correct_uses_range_slice(gold) -> None:
    _, corr = poly(
        gold,
        angle_range=(10.0, -10.0),
        eV_range=(0.2, -0.2),
        temp=100.0,
        use_step_edge=True,
        vary_temp=False,
        degree=2,
        plot=False,
        correct=True,
        crop_correct=True,
        parallel_kw={"backend": "threading", "return_as": "list"},
    )

    assert corr.alpha.min() >= -10.0
    assert corr.alpha.max() <= 10.0
    assert corr.alpha.size < gold.alpha.size
    assert corr.eV.size < gold.eV.size


def test_spline_crop_correct_uses_range_slice(gold) -> None:
    _, corr = spline(
        gold,
        angle_range=(10.0, -10.0),
        eV_range=(0.2, -0.2),
        temp=100.0,
        use_step_edge=True,
        vary_temp=False,
        plot=False,
        correct=True,
        crop_correct=True,
        parallel_kw={"backend": "threading", "return_as": "list"},
    )

    assert corr.alpha.min() >= -10.0
    assert corr.alpha.max() <= 10.0
    assert corr.alpha.size < gold.alpha.size
    assert corr.eV.size < gold.eV.size


def test_edge_fixed_center_fixes_center_parameter(gold) -> None:
    ds = edge(
        gold,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        temp=100.0,
        vary_temp=False,
        fixed_center=0.2,
        normalize=False,
        bkg_slope=True,
        return_full=True,
        progress=False,
        parallel_kw={"backend": "threading", "n_jobs": 1, "return_as": "list"},
    )
    center_coeff = ds.modelfit_coefficients.sel(param="center").values
    finite = np.isfinite(center_coeff)
    assert finite.any()
    assert_allclose(center_coeff[finite], 0.2, atol=1e-12)

    first = ds.modelfit_results.isel(alpha=0).item()
    assert first.params["center"].value == 0.2
    assert first.params["center"].vary is False
    assert first.params["back1"].vary is True


def test_edge_fixed_center_with_normalize_sets_normalized_parameter(gold) -> None:
    angle_range = (-15, 15)
    eV_range = (-0.2, 0.2)
    fixed_center = 0.04

    gold_sel = gold.sel(alpha=slice(*angle_range), eV=slice(*eV_range))
    avgx = float(gold_sel.eV.values.mean())
    stdx = float(gold_sel.eV.values.std())
    expected_center = (fixed_center - avgx) / stdx

    ds = edge(
        gold,
        angle_range=angle_range,
        eV_range=eV_range,
        temp=100.0,
        vary_temp=False,
        fixed_center=fixed_center,
        normalize=True,
        bkg_slope=True,
        return_full=True,
        progress=False,
        parallel_kw={"backend": "threading", "n_jobs": 1, "return_as": "list"},
    )
    center_coeff = ds.modelfit_coefficients.sel(param="center").values
    finite = np.isfinite(center_coeff)
    assert finite.any()
    assert_allclose(center_coeff[finite], expected_center, atol=1e-12)
    assert_allclose(center_coeff[finite] * stdx + avgx, fixed_center, atol=1e-12)

    first = ds.modelfit_results.isel(alpha=0).item()
    assert_allclose(first.params["center"].value, expected_center, atol=1e-12)
    assert first.params["center"].vary is False


def test_edge_fixed_center_with_normalize_returns_physical_center(gold) -> None:
    vals, _errs = typing.cast(
        "tuple[xr.DataArray, xr.DataArray]",
        edge(
            gold,
            angle_range=(-15, 15),
            eV_range=(-0.2, 0.2),
            temp=100.0,
            vary_temp=False,
            fixed_center=0.04,
            normalize=True,
            bkg_slope=True,
            return_full=False,
            progress=False,
            parallel_kw={"backend": "threading", "n_jobs": 1, "return_as": "list"},
        ),
    )
    finite = np.isfinite(vals.values)
    assert finite.any()
    assert_allclose(vals.values[finite], 0.04, atol=1e-12)


def test_edge_normalize_scales_energy_parameters(
    gold: xr.DataArray, monkeypatch: pytest.MonkeyPatch
) -> None:
    angle_range = (-15, 15)
    eV_range = (-0.2, 0.2)
    temp = 100.0
    resolution = 0.02
    gold_sel = gold.sel(alpha=slice(*angle_range), eV=slice(*eV_range))
    stdx = float(gold_sel.eV.values.std())
    received: dict[str, float] = {}
    original = gold_mod._edge_model_and_params

    def capture_energy_parameters(**kwargs: typing.Any):
        received["temp"] = kwargs["temp"]
        received["resolution"] = kwargs["resolution"]
        return original(**kwargs)

    monkeypatch.setattr(gold_mod, "_edge_model_and_params", capture_energy_parameters)

    edge(
        gold,
        angle_range=angle_range,
        eV_range=eV_range,
        temp=temp,
        resolution=resolution,
        normalize=True,
        progress=False,
        parallel_kw={"backend": "threading", "n_jobs": 1, "return_as": "list"},
    )

    assert received["temp"] == pytest.approx(temp / stdx)
    assert received["resolution"] == pytest.approx(resolution / stdx)


@pytest.mark.parametrize("adaptive", [False, True], ids=["fixed", "adaptive"])
def test_edge_is_invariant_to_energy_offset(gold: xr.DataArray, adaptive: bool) -> None:
    offset = 71.0
    kinetic = gold.assign_coords(eV=gold.eV + offset)
    shifted = kinetic.assign_coords(eV=kinetic.eV - offset)
    kwargs = {
        "angle_range": (-15, 15),
        "adaptive": adaptive,
        "temp": 100.0,
        "resolution": 0.02,
        "use_step_edge": True,
        "normalize": True,
        "progress": False,
        "parallel_kw": {
            "backend": "threading",
            "n_jobs": 1,
            "return_as": "list",
        },
    }

    kinetic_center, kinetic_stderr = typing.cast(
        "tuple[xr.DataArray, xr.DataArray]",
        edge(kinetic, eV_range=(70.8, 71.2), **kwargs),
    )
    shifted_center, shifted_stderr = typing.cast(
        "tuple[xr.DataArray, xr.DataArray]",
        edge(shifted, eV_range=(-0.2, 0.2), **kwargs),
    )

    assert_allclose(kinetic_center - offset, shifted_center, atol=1e-8)
    assert_allclose(kinetic_stderr, shifted_stderr, atol=1e-8)


def test_edge_range_selection_follows_descending_coordinate_order(gold) -> None:
    gold_desc = gold.isel(alpha=slice(None, None, -1), eV=slice(None, None, -1))
    gold_desc = gold_desc.assign_coords(
        alpha=gold.alpha.values[::-1], eV=gold.eV.values[::-1]
    )

    vals, _errs = typing.cast(
        "tuple[xr.DataArray, xr.DataArray]",
        edge(
            gold_desc,
            along="alpha",
            angle_range=(-15, 15),
            eV_range=(-0.2, 0.2),
            temp=100.0,
            vary_temp=False,
            fixed_center=0.04,
            normalize=False,
            bkg_slope=True,
            return_full=False,
            progress=False,
            parallel_kw={"backend": "threading", "n_jobs": 1, "return_as": "list"},
        ),
    )
    finite = np.isfinite(vals.values)
    assert finite.any()
    assert_allclose(vals.values[finite], 0.04, atol=1e-12)


def test_poly_plot_supports_open_bounds(gold) -> None:
    fig = plt.figure()

    res = poly(
        gold,
        angle_range=(None, 15.0),
        eV_range=(-0.2, None),
        temp=100.0,
        vary_temp=False,
        degree=2,
        use_step_edge=True,
        plot=True,
        fig=fig,
        parallel_kw={"backend": "threading", "n_jobs": 1, "return_as": "list"},
    )

    assert isinstance(res, xr.Dataset)

    plt.close(fig)


def test_quick_fit_plot_fwhm_span_matches_resolution(gold) -> None:
    fig, ax = plt.subplots()
    ds = quick_fit(
        gold,
        eV_range=(-0.2, 0.2),
        temp=100.0,
        resolution=1e-2,
        fix_temp=True,
        fix_resolution=True,
        plot=True,
        ax=ax,
    )
    resolution = float(ds.modelfit_coefficients.sel(param="resolution"))

    assert len(ax.patches) > 0
    span = ax.patches[-1]
    assert_allclose(span.get_width(), resolution, atol=1e-12)

    plt.close(fig)


def test_quick_fit_descending_eV_range_selection_matches_ascending(gold) -> None:
    gold_desc = gold.isel(eV=slice(None, None, -1)).assign_coords(eV=gold.eV[::-1])

    asc = quick_fit(
        gold,
        eV_range=(-0.2, 0.2),
        temp=100.0,
        resolution=1e-2,
        fix_temp=True,
        fix_resolution=True,
        plot=False,
    )
    desc = quick_fit(
        gold_desc,
        eV_range=(-0.2, 0.2),
        temp=100.0,
        resolution=1e-2,
        fix_temp=True,
        fix_resolution=True,
        plot=False,
    )

    assert desc.modelfit_data.sizes["eV"] == asc.modelfit_data.sizes["eV"]

    center = float(desc.modelfit_coefficients.sel(param="center"))
    resolution = float(desc.modelfit_coefficients.sel(param="resolution"))
    assert np.isfinite(center)
    assert np.isfinite(resolution)
    assert -0.2 <= center <= 0.2
    assert resolution > 0.0


@pytest.mark.parametrize("bkg_slope", [True, False], ids=["slope", "no_slope"])
@pytest.mark.parametrize("fix_resolution", [False, True], ids=["fix_res", "vary_res"])
@pytest.mark.parametrize("fix_center", [False, True], ids=["fix_center", "vary_center"])
@pytest.mark.parametrize("fix_temp", [True, False], ids=["fix_temp", "vary_temp"])
@pytest.mark.parametrize("resolution", [None, 1e-2], ids=["res_None", "res_1e-2"])
@pytest.mark.parametrize("temp", [None, 100.0], ids=["temp_None", "temp_100"])
@pytest.mark.parametrize("eV_range", [None, (-0.2, 0.2)], ids=["eV_full", "eV_range"])
def test_quick_fit(
    gold, eV_range, temp, resolution, fix_temp, fix_center, fix_resolution, bkg_slope
) -> None:
    ds = quick_fit(
        gold,
        eV_range=eV_range,
        temp=temp,
        resolution=resolution,
        fix_temp=fix_temp,
        fix_center=fix_center,
        fix_resolution=fix_resolution,
        bkg_slope=bkg_slope,
        plot=True,
    )
    plt.close()
    assert ds.modelfit_results.item().success
