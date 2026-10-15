import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def linear_fit_result():
    import lmfit.models

    x = np.linspace(-2.0, 2.0, 21)
    y = 2.5 * x + 0.7 + 0.12 * np.sin(np.arange(x.size))
    y[[0, 9, -1]] = np.nan
    data = xr.DataArray(y, dims="energy", coords={"energy": x})
    return data.xlm.modelfit(
        "energy", model=lmfit.models.LinearModel(), nan_policy="omit"
    )


def test_plot_fit_native_without_model_result(linear_fit_result):
    from erlab.plotting import FitArtists, plot_fit

    result = linear_fit_result.drop_vars("modelfit_results")
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax)
        assert isinstance(artists, FitArtists)
        np.testing.assert_equal(artists.data.get_xdata(), result.energy.values)
        np.testing.assert_equal(artists.data.get_ydata(), result.modelfit_data.values)
        np.testing.assert_equal(
            artists.fit.get_ydata(), result.modelfit_best_fit.values
        )
        assert artists.data.get_linestyle() == "None"
        assert artists.band is None
        assert artists.components == {}
        assert artists.component_bands == {}
    finally:
        plt.close(fig)


@pytest.mark.parametrize("sigma", [2, 0.95])
@pytest.mark.parametrize("numpoints", [None, 81])
def test_plot_fit_uncertainty_delegation_and_isolation(
    linear_fit_result, monkeypatch, sigma, numpoints
):
    import lmfit.model

    from erlab.plotting import plot_fit

    result = linear_fit_result
    model_result = result.modelfit_results.item()
    parameters_before = model_result.params.valuesdict()
    cache_before = {
        key: getattr(model_result, key, None)
        for key in ("dely", "dely_predicted", "dely_comps")
    }
    calls = []

    def eval_uncertainty(self, *, params, sigma, **kwargs):
        calls.append((self, params, sigma, kwargs))
        self.dely = self.dely_predicted = self.dely_comps = object()
        params["slope"].set(value=999.0)
        return np.full(np.asarray(kwargs["x"]).shape, 0.25)

    monkeypatch.setattr(lmfit.model.ModelResult, "eval_uncertainty", eval_uncertainty)
    valid = np.isfinite(result.modelfit_best_fit.values)
    x = result.energy.values
    fitted = result.modelfit_best_fit.values
    if numpoints is not None:
        x = np.linspace(x[valid].min(), x[valid].max(), numpoints)
        fitted = model_result.eval(x=x)
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax, numpoints=numpoints, sigma=sigma)
        evaluated_result, evaluated_params, evaluated_sigma, kwargs = calls.pop()
        assert not calls
        assert evaluated_result is not model_result
        assert evaluated_params is not model_result.params
        assert evaluated_sigma == sigma
        np.testing.assert_equal(kwargs["x"], x if numpoints else x[valid])
        np.testing.assert_equal(artists.fit.get_xdata(), x)
        np.testing.assert_equal(artists.fit.get_ydata(), fitted)
        vertices = np.concatenate([path.vertices for path in artists.band.get_paths()])
        band_valid = np.isfinite(fitted)
        for xi, yi in zip(x[band_valid], fitted[band_valid], strict=True):
            bounds = vertices[vertices[:, 0] == xi, 1]
            np.testing.assert_allclose(
                [bounds.min(), bounds.max()], [yi - 0.25, yi + 0.25]
            )
        if numpoints is None:
            assert len(artists.band.get_paths()) == 2
        assert artists.band.get_zorder() < artists.fit.get_zorder()
        assert model_result.params.valuesdict() == parameters_before
        for key, value in cache_before.items():
            assert getattr(model_result, key, None) is value
    finally:
        plt.close(fig)


@pytest.fixture
def auxiliary_coordinate_fit_result():
    import lmfit.models

    energy = np.linspace(-1.0, 1.0, 11)
    y = 2.0 * energy + 1.0 + 0.02 * np.sin(np.arange(energy.size))
    y[[0, 5, -1]] = np.nan
    data = xr.DataArray(
        y,
        dims="sample",
        coords={"sample": np.arange(energy.size), "energy": ("sample", energy)},
    )
    return data.xlm.modelfit(
        "energy",
        reduce_dims=["sample"],
        model=lmfit.models.LinearModel(),
        nan_policy="omit",
    )


@pytest.mark.parametrize("coord", [None, "energy"])
def test_plot_fit_auxiliary_coordinate_native_without_result(
    auxiliary_coordinate_fit_result, coord
):
    from erlab.plotting import plot_fit

    result = auxiliary_coordinate_fit_result.drop_vars("modelfit_results")
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax, coord=coord)
        x = result["sample" if coord is None else coord].values
        np.testing.assert_equal(artists.data.get_xdata(), x)
        np.testing.assert_equal(artists.fit.get_xdata(), x)
        np.testing.assert_equal(artists.fit.get_ydata(), result.modelfit_best_fit)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("coord", [None, "energy"])
def test_plot_fit_auxiliary_coordinate_native_uncertainty(
    auxiliary_coordinate_fit_result, monkeypatch, coord
):
    import lmfit.model

    from erlab.plotting import plot_fit

    result = auxiliary_coordinate_fit_result
    # Plot coordinates must not replace the original independent input.
    x = result["sample" if coord is None else coord].values.astype(float).copy()
    x[3] = np.nan
    result = result.assign_coords({"sample" if coord is None else coord: ("sample", x)})
    model_result = result.modelfit_results.item()
    input_before = model_result.userkws["x"].copy()
    calls = []

    def eval_uncertainty(self, *, params, sigma, **kwargs):
        calls.append(kwargs)
        return np.full(np.asarray(kwargs["x"]).shape, 0.25)

    monkeypatch.setattr(lmfit.model.ModelResult, "eval_uncertainty", eval_uncertainty)
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax, coord=coord, sigma=0.95)
        np.testing.assert_equal(calls.pop()["x"], input_before)
        assert not calls
        np.testing.assert_equal(artists.fit.get_xdata(), x)
        np.testing.assert_equal(artists.fit.get_ydata(), result.modelfit_best_fit)
        vertices = np.concatenate([path.vertices for path in artists.band.get_paths()])
        assert np.all(np.isfinite(vertices))
        np.testing.assert_equal(model_result.userkws["x"], input_before)
    finally:
        plt.close(fig)


def test_plot_fit_auxiliary_coordinate_dense_grid(auxiliary_coordinate_fit_result):
    from erlab.plotting import plot_fit

    result = auxiliary_coordinate_fit_result
    model_result = result.modelfit_results.item()
    input_before = model_result.userkws["x"].copy()
    dense_x = np.linspace(input_before.min(), input_before.max(), 51)
    expected = model_result.eval(x=dense_x)
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="Set coord"):
            plot_fit(result, ax, numpoints=51)
        assert not ax.lines
        assert not ax.collections
        artists = plot_fit(result, ax, coord="energy", numpoints=51)
        np.testing.assert_allclose(artists.fit.get_xdata(), dense_x)
        np.testing.assert_allclose(artists.fit.get_ydata(), expected)
        np.testing.assert_equal(artists.data.get_xdata(), result.energy.values)
        np.testing.assert_equal(model_result.userkws["x"], input_before)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("show_yerr", [False, True])
def test_plot_fit_data_style_aliases_override_defaults(linear_fit_result, show_yerr):
    from erlab.plotting import plot_fit

    result = linear_fit_result.assign(modelfit_weights=10.0)
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(
            result,
            ax,
            show_yerr=show_yerr,
            data_kws={"ls": "--", "c": "black", "ms": 4},
            fit_kws={"ls": "-.", "c": "blue", "ms": 6},
        )
        line = artists.data.lines[0] if show_yerr else artists.data
        assert line.get_linestyle() == "--"
        assert line.get_color() == "black"
        assert line.get_markersize() == 4
        assert artists.fit.get_linestyle() == "-."
        assert artists.fit.get_color() == "blue"
        assert artists.fit.get_markersize() == 6
        assert len(ax.collections) == int(show_yerr)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("weighted", [False, True])
def test_plot_fit_hidden_error_styles_preserve_data_line(linear_fit_result, weighted):
    from erlab.plotting import plot_fit

    result = linear_fit_result
    if weighted:
        result = result.assign(modelfit_weights=10.0)
    data_kws = {
        "color": "purple",
        "marker": "s",
        "ecolor": "black",
        "elinewidth": 2.0,
        "capsize": 4.0,
        "capthick": 1.0,
        "barsabove": True,
        "errorevery": 2,
    }
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax, show_yerr=not weighted, data_kws=data_kws)
        assert artists.data.get_color() == "purple"
        assert artists.data.get_marker() == "s"
        assert not ax.collections
        assert data_kws["ecolor"] == "black"
        assert data_kws["capsize"] == 4.0
    finally:
        plt.close(fig)


def test_plot_fit_format_strings(linear_fit_result):
    from erlab.plotting import plot_fit

    fig, ax = plt.subplots()
    try:
        artists = plot_fit(linear_fit_result, ax, datafmt="rs-", fitfmt="g^:")
        assert artists.data.get_marker() == "s"
        assert artists.data.get_linestyle() == "-"
        assert artists.data.get_color() == "r"
        assert artists.fit.get_marker() == "^"
        assert artists.fit.get_linestyle() == ":"
        assert artists.fit.get_color() == "g"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "band_kws",
    [
        {"facecolor": "red", "edgecolor": "purple"},
        {"fc": "red", "ec": "purple"},
    ],
)
def test_plot_fit_band_colors_override_defaults(linear_fit_result, band_kws):
    from erlab.plotting import plot_fit

    fig, ax = plt.subplots()
    try:
        artists = plot_fit(linear_fit_result, ax, sigma=0.95, band_kws=band_kws)
        np.testing.assert_allclose(
            artists.band.get_facecolor()[0], matplotlib.colors.to_rgba("red", 0.2)
        )
        np.testing.assert_allclose(
            artists.band.get_edgecolor()[0], matplotlib.colors.to_rgba("purple", 0.2)
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("numpoints", [None, 81])
def test_plot_fit_weight_errors_preserve_data_grid(linear_fit_result, numpoints):
    import lmfit.models

    from erlab.plotting import plot_fit

    data = linear_fit_result.modelfit_data
    weights = xr.DataArray(
        np.linspace(2.0, 4.0, data.size), dims=data.dims, coords=data.coords
    )
    result = data.xlm.modelfit(
        "energy", model=lmfit.models.LinearModel(), weights=weights, nan_policy="omit"
    )
    if numpoints is None:
        result = result.drop_vars("modelfit_results")
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax, numpoints=numpoints)
        np.testing.assert_equal(artists.data.lines[0].get_xdata(), data.energy)
        np.testing.assert_equal(artists.data.lines[0].get_ydata(), data)
        segments = artists.data.lines[2][0].get_segments()
        for i, segment in enumerate(segments):
            if np.isfinite(data.values[i]):
                np.testing.assert_allclose(
                    segment,
                    [
                        [
                            data.energy.values[i],
                            data.values[i] - 1.0 / weights.values[i],
                        ],
                        [
                            data.energy.values[i],
                            data.values[i] + 1.0 / weights.values[i],
                        ],
                    ],
                )
            else:
                assert segment.size == 0
        assert artists.fit.get_xdata().size == (data.size if numpoints is None else 81)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "weights", [4.0, np.array([4.0, 0.0, np.nan, np.inf, -1.0] + [4.0] * 16)]
)
def test_plot_fit_weight_errors_broadcast_and_skip_invalid(linear_fit_result, weights):
    from erlab.plotting import plot_fit

    result = linear_fit_result.drop_vars("modelfit_results").assign(
        modelfit_weights=("energy", weights) if np.ndim(weights) else weights
    )
    weights = np.broadcast_to(weights, result.modelfit_data.shape)
    data = result.modelfit_data.values
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax)
        for i, segment in enumerate(artists.data.lines[2][0].get_segments()):
            if np.isfinite(data[i]) and np.isfinite(weights[i]) and weights[i] > 0.0:
                np.testing.assert_allclose(
                    segment[:, 1], data[i] + np.array([-1, 1]) / weights[i]
                )
            else:
                assert segment.size == 0
    finally:
        plt.close(fig)


def test_plot_fit_visibility(linear_fit_result):
    from erlab.plotting import plot_fit

    fig, ax = plt.subplots()
    try:
        artists = plot_fit(linear_fit_result, ax, show_fit=False)
        assert artists.fit is None
        assert artists.band is None
        hidden = plot_fit(
            linear_fit_result, ax, show_data=False, show_fit=False, sigma=0.95
        )
        assert hidden.data is None
        assert hidden.fit is None
        assert hidden.band is not None
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "options",
    [
        {"numpoints": 1},
        {"numpoints": 2.5},
        {"sigma": 0},
        {"sigma": -1},
        {"sigma": np.nan},
        {"sigma": np.inf},
        {"sigma": -np.inf},
        {"coord": "missing"},
        {"coord": "param"},
    ],
)
def test_plot_fit_invalid_options_do_not_add_artists(linear_fit_result, options):
    from erlab.plotting import plot_fit

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"numpoints|sigma|coord"):
            plot_fit(linear_fit_result, ax, **options)
        assert not ax.lines
        assert not ax.collections
        assert not ax.containers
    finally:
        plt.close(fig)


@pytest.mark.parametrize("missing", ["result", "covariance", "stderr"])
def test_plot_fit_unavailable_uncertainty(linear_fit_result, missing):
    from erlab.plotting import plot_fit

    result = linear_fit_result
    model_result = result.modelfit_results.item()
    if missing == "result":
        result = result.drop_vars("modelfit_results")
    elif missing == "covariance":
        model_result.covar = None
    else:
        model_result.params["slope"].stderr = None
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"ModelResult|covariance"):
            plot_fit(result, ax, sigma=0.95)
        assert not ax.lines
        assert not ax.collections
    finally:
        plt.close(fig)


def test_plot_fit_dataset_prefixes_and_singletons(linear_fit_result):
    from erlab.plotting import plot_fit

    def prefixed(name):
        result = linear_fit_result.assign(
            modelfit_weights=2.0 if name == "first" else 4.0
        )
        return result.rename({str(key): f"{name}_{key}" for key in result.data_vars})

    result = xr.merge([prefixed("first"), prefixed("second")]).expand_dims(scan=[0])
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="data_var"):
            plot_fit(result, ax)
        artists = plot_fit(result, ax, data_var="first")
        np.testing.assert_equal(
            artists.fit.get_ydata(), linear_fit_result.modelfit_best_fit
        )
        segment = artists.data.lines[2][0].get_segments()[1]
        np.testing.assert_allclose(
            segment[:, 1], linear_fit_result.modelfit_data.values[1] + [-0.5, 0.5]
        )
        with pytest.raises(ValueError, match="No fitted"):
            plot_fit(result, ax, data_var="missing")
        stack = xr.concat([linear_fit_result, linear_fit_result], dim="scan")
        with pytest.raises(ValueError, match="single one-dimensional"):
            plot_fit(stack, ax)
    finally:
        plt.close(fig)


def test_plot_fit_sampled_auxiliary_rejects_dense_grid():
    import lmfit

    from erlab.plotting import plot_fit

    def sampled_model(x, y, scale=1.0, offset=0.0):
        return scale * np.asarray(y) + offset + np.asarray(x)

    x = np.linspace(0, 2, 21)
    y = 2 * x + np.sin(x)
    y[[0, 9, -1]] = np.nan
    data = xr.DataArray(y, dims="energy", coords={"energy": x})
    result = data.xlm.modelfit(
        "energy",
        model=lmfit.Model(sampled_model, independent_vars=["x", "y"]),
        guess=False,
        nan_policy="omit",
    )
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="Sampled auxiliary"):
            plot_fit(result, ax, numpoints=51)
        assert not ax.lines
        assert not ax.collections
        artists = plot_fit(result, ax)
        np.testing.assert_equal(artists.fit.get_ydata(), result.modelfit_best_fit)
    finally:
        plt.close(fig)


def test_plot_fit_dense_curve_evaluates_nonlinear_model():
    import lmfit.models

    from erlab.plotting import plot_fit

    x = np.linspace(-3.0, 3.0, 17)
    y = 3.0 * np.exp(-(((x - 0.3) / 0.7) ** 2)) + 0.03 * np.sin(np.arange(x.size))
    result = xr.DataArray(y, dims="x", coords={"x": x}).xlm.modelfit(
        "x", model=lmfit.models.GaussianModel()
    )
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax, numpoints=101)
        dense_x = artists.fit.get_xdata()
        expected = result.modelfit_results.item().eval(x=dense_x)
        np.testing.assert_allclose(artists.fit.get_ydata(), expected)
        interpolated = np.interp(dense_x, x, result.modelfit_best_fit.values)
        assert np.max(np.abs(expected - interpolated)) > 0.05
    finally:
        plt.close(fig)


@pytest.fixture
def composite_fit_result():
    import lmfit.models

    model = lmfit.models.GaussianModel(prefix="peak_") + lmfit.models.LinearModel(
        prefix="bg_"
    )
    params = model.make_params(
        peak_amplitude=2.0,
        peak_center=0.2,
        peak_sigma=0.4,
        bg_slope=0.1,
        bg_intercept=0.2,
    )
    x = np.linspace(-2.0, 2.0, 81)
    y = model.eval(params=params, x=x) + 0.03 * np.sin(np.arange(x.size))
    y[[0, 35, -1]] = np.nan
    return xr.DataArray(y, dims="x", coords={"x": x}).xlm.modelfit(
        "x", model=model, params=params, guess=False, nan_policy="omit"
    )


@pytest.mark.parametrize("numpoints", [None, 101])
def test_plot_fit_components_preserve_grid_and_model(composite_fit_result, numpoints):
    from erlab.plotting import plot_fit

    result = composite_fit_result
    model_result = result.modelfit_results.item()
    valid = np.isfinite(result.modelfit_best_fit.values)
    x = result.x.values
    if numpoints is None:
        expected = model_result.eval_components()
    else:
        x = np.linspace(x[valid].min(), x[valid].max(), numpoints)
        expected = model_result.eval_components(x=x)
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax, show_components=True, numpoints=numpoints)
        assert set(artists.components) == {"peak_", "bg_"}
        assert artists.component_bands == {}
        for name, values in expected.items():
            line = artists.components[name]
            np.testing.assert_equal(line.get_xdata(), x)
            if numpoints is None:
                native = np.full(x.shape, np.nan)
                native[valid] = values
                values = native
            np.testing.assert_allclose(line.get_ydata(), values)
            assert line.get_label() == name
            assert line.get_linestyle() == "--"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("numpoints", [None, 101])
@pytest.mark.parametrize(
    "band_kws",
    [
        None,
        {
            "fc": "red",
            "ec": "purple",
            "alpha": 0.4,
            "zorder": -1,
            "label": "Total interval",
        },
    ],
)
def test_plot_fit_component_uncertainty_delegation(
    composite_fit_result, monkeypatch, numpoints, band_kws
):
    import lmfit.model

    from erlab.plotting import plot_fit

    result = composite_fit_result
    model_result = result.modelfit_results.item()
    parameters_before = model_result.params.valuesdict()
    cache_before = {
        key: getattr(model_result, key, None)
        for key in ("dely", "dely_predicted", "dely_comps")
    }
    widths = {"peak_": 0.4, "bg_": 0.07}
    calls = []

    def eval_uncertainty(self, *, params, sigma, **kwargs):
        calls.append((self, params, sigma, kwargs))
        shape = np.asarray(kwargs["x"]).shape
        self.dely = self.dely_predicted = np.full(shape, 0.25)
        self.dely_comps = {
            name: np.full(shape, width) for name, width in widths.items()
        }
        params["peak_amplitude"].set(value=999.0)
        return self.dely

    monkeypatch.setattr(lmfit.model.ModelResult, "eval_uncertainty", eval_uncertainty)
    valid = np.isfinite(result.modelfit_best_fit.values)
    expected_x = result.x.values
    if numpoints is not None:
        expected_x = np.linspace(
            expected_x[valid].min(), expected_x[valid].max(), numpoints
        )
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(
            result,
            ax,
            show_components=True,
            numpoints=numpoints,
            sigma=0.95,
            component_kws={"peak_": {"c": "purple"}, "bg_": {"color": "black"}},
            band_kws=band_kws,
        )
        evaluated_result, evaluated_params, sigma, kwargs = calls.pop()
        assert not calls
        assert evaluated_result is not model_result
        assert evaluated_params is not model_result.params
        assert sigma == 0.95
        np.testing.assert_equal(
            kwargs["x"], expected_x[valid] if numpoints is None else expected_x
        )
        assert artists.component_bands.keys() == artists.components.keys()
        for name, band in artists.component_bands.items():
            line = artists.components[name]
            np.testing.assert_equal(line.get_xdata(), expected_x)
            vertices = np.concatenate([path.vertices for path in band.get_paths()])
            finite = np.isfinite(line.get_ydata())
            for xi, yi in zip(
                expected_x[finite], line.get_ydata()[finite], strict=True
            ):
                bounds = vertices[vertices[:, 0] == xi, 1]
                np.testing.assert_allclose(
                    [bounds.min(), bounds.max()], [yi - widths[name], yi + widths[name]]
                )
            if numpoints is None:
                assert len(band.get_paths()) == 2
            if band_kws is None:
                np.testing.assert_allclose(
                    band.get_facecolor()[0],
                    matplotlib.colors.to_rgba(line.get_color(), 0.2),
                )
                assert band.get_zorder() < line.get_zorder()
        if band_kws is not None:
            for band in [artists.band, *artists.component_bands.values()]:
                np.testing.assert_allclose(
                    band.get_facecolor()[0], matplotlib.colors.to_rgba("red", 0.4)
                )
                np.testing.assert_allclose(
                    band.get_edgecolor()[0], matplotlib.colors.to_rgba("purple", 0.4)
                )
                assert band.get_zorder() == -1
            assert ax.get_legend_handles_labels()[1].count("Total interval") == 1
        assert ax.get_legend_handles_labels()[1].count("peak_") == 1
        assert ax.get_legend_handles_labels()[1].count("bg_") == 1
        assert model_result.params.valuesdict() == parameters_before
        for key, value in cache_before.items():
            assert getattr(model_result, key, None) is value
    finally:
        plt.close(fig)


def test_plot_fit_component_styles_and_visibility(composite_fit_result):
    from erlab.plotting import plot_fit

    component_kws = {
        "peak_": {"c": "purple", "ls": ":", "label": "Peak", "lw": 3},
        "bg_": {"color": "black", "linestyle": "-.", "alpha": 0.5},
    }
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(
            composite_fit_result,
            ax,
            show_components=True,
            show_data=False,
            show_fit=False,
            component_kws=component_kws,
        )
        assert artists.data is None
        assert artists.fit is None
        assert artists.band is None
        peak = artists.components["peak_"]
        assert peak.get_color() == "purple"
        assert peak.get_linestyle() == ":"
        assert peak.get_label() == "Peak"
        assert peak.get_linewidth() == 3
        background = artists.components["bg_"]
        assert background.get_color() == "black"
        assert background.get_linestyle() == "-."
        assert background.get_alpha() == 0.5
        assert len(ax.lines) == 2
        assert component_kws["peak_"]["ls"] == ":"
    finally:
        plt.close(fig)


def test_plot_fit_component_band_styles_and_labels(composite_fit_result, monkeypatch):
    import copy

    import lmfit.model

    from erlab.plotting import plot_fit

    def eval_uncertainty(self, **kwargs):
        delta = np.full(np.asarray(kwargs["x"]).shape, 0.1)
        self.dely_comps = {"peak_": delta, "bg_": delta}
        return delta

    monkeypatch.setattr(lmfit.model.ModelResult, "eval_uncertainty", eval_uncertainty)
    band_kws = {"label": "Total interval", "alpha": 0.3, "hatch": "//"}
    component_band_kws = {
        "peak_": {"label": "Peak interval", "fc": "green", "alpha": 0.5},
        "bg_": {"label": "Background interval", "hatch": "xx"},
    }
    before = copy.deepcopy(component_band_kws)
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(
            composite_fit_result,
            ax,
            show_components=True,
            sigma=0.95,
            component_kws={"bg_": {"color": "black"}},
            band_kws=band_kws,
            component_band_kws=component_band_kws,
        )
        peak = artists.component_bands["peak_"]
        background = artists.component_bands["bg_"]
        assert artists.band.get_label() == "Total interval"
        assert peak.get_label() == "Peak interval"
        assert background.get_label() == "Background interval"
        assert peak.get_hatch() == "//"
        assert background.get_hatch() == "xx"
        np.testing.assert_allclose(
            peak.get_facecolor()[0], matplotlib.colors.to_rgba("green", 0.5)
        )
        np.testing.assert_allclose(
            background.get_facecolor()[0], matplotlib.colors.to_rgba("black", 0.3)
        )
        assert ax.get_legend_handles_labels()[1].count("Total interval") == 1
        assert component_band_kws == before
        assert band_kws == {"label": "Total interval", "alpha": 0.3, "hatch": "//"}
    finally:
        plt.close(fig)


@pytest.mark.parametrize("style_arg", ["component_kws", "component_band_kws"])
def test_plot_fit_components_reject_unknown_styles(composite_fit_result, style_arg):
    from erlab.plotting import plot_fit

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"Unknown model component names.*'peak'"):
            plot_fit(
                composite_fit_result,
                ax,
                show_components=True,
                **{style_arg: {"peak": {"color": "purple"}}},
            )
        assert not ax.lines
        assert not ax.collections
        hidden = plot_fit(
            composite_fit_result, ax, **{style_arg: {"peak": {"color": "purple"}}}
        )
        assert hidden.components == {}
    finally:
        plt.close(fig)


def test_plot_fit_components_require_retained_result(composite_fit_result):
    from erlab.plotting import plot_fit

    result = composite_fit_result.drop_vars("modelfit_results")
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="retained lmfit ModelResult"):
            plot_fit(result, ax, show_components=True)
        assert not ax.lines
        assert not ax.collections
    finally:
        plt.close(fig)


def test_plot_fit_native_components_use_original_auxiliary_input(
    auxiliary_coordinate_fit_result,
):
    from erlab.plotting import plot_fit

    result = auxiliary_coordinate_fit_result
    x = result["sample"].values.astype(float)
    x[3] = np.nan
    result = result.assign_coords(sample=x)
    model_result = result.modelfit_results.item()
    expected = model_result.eval_components()
    valid = np.isfinite(result.modelfit_best_fit.values)
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax, show_components=True)
        for name, values in expected.items():
            line = artists.components[name]
            native = np.full(x.shape, np.nan)
            native[valid] = values
            np.testing.assert_equal(line.get_xdata(), x)
            np.testing.assert_allclose(line.get_ydata(), native)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("numpoints", [None, 121])
def test_plot_fit_multipeak_components_use_model_evaluation(numpoints):
    import erlab.analysis as era
    from erlab.plotting import plot_fit

    model = era.fit.models.MultiPeakModel(
        npeaks=1,
        peak_shapes=["gaussian"],
        fd=True,
        background="constant",
        convolve=True,
    )
    params = model.make_params(
        p0_height=2.0,
        p0_center=-0.1,
        p0_width=0.2,
        const_bkg=0.1,
        temp=30.0,
        efermi=0.0,
        resolution=0.02,
    )
    x = np.linspace(-0.5, 0.5, 101)
    y = model.eval(params=params, x=x) + 0.01 * np.sin(np.arange(x.size))
    result = xr.DataArray(y, dims="x", coords={"x": x}).xlm.modelfit(
        "x", model=model, params=params, guess=False
    )
    plot_x = x if numpoints is None else np.linspace(x.min(), x.max(), numpoints)
    expected = result.modelfit_results.item().eval_components(x=plot_x)
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(result, ax, show_components=True, numpoints=numpoints)
        assert artists.components.keys() == expected.keys()
        assert any(name.endswith("_fd") for name in expected)
        for name, values in expected.items():
            np.testing.assert_equal(artists.components[name].get_xdata(), plot_x)
            np.testing.assert_allclose(artists.components[name].get_ydata(), values)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("numpoints", [None, 31])
def test_plot_fit_scalar_component_broadcasts(
    linear_fit_result, monkeypatch, numpoints
):
    from erlab.plotting import plot_fit

    model_result = linear_fit_result.modelfit_results.item()
    monkeypatch.setattr(
        model_result.model, "eval_components", lambda **_: {"offset": 1.0}
    )
    fig, ax = plt.subplots()
    try:
        artists = plot_fit(
            linear_fit_result, ax, show_components=True, numpoints=numpoints
        )
        line = artists.components["offset"]
        if numpoints is None:
            expected = np.where(
                np.isfinite(linear_fit_result.modelfit_best_fit.values), 1.0, np.nan
            )
        else:
            expected = np.ones(numpoints)
        np.testing.assert_equal(line.get_ydata(), expected)
    finally:
        plt.close(fig)
