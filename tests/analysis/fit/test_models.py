import lmfit
import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.analysis.fit import models
from erlab.analysis.fit.functions.general import sc_spectral_function, tll, voigt


def test_fermi_dirac_model() -> None:
    # Create a test input DataArray
    x = np.linspace(-0.1, 0.1, 100)
    data = np.zeros_like(x)

    # Create an instance of FermiDiracModel
    model = models.FermiDiracModel()

    # Set initial parameter values
    params = model.guess(data, x=x)
    params["center"].set(value=0.0)
    params["temp"].set(value=30.0)
    params["resolution"].set(value=0.02)

    # Generate test data based on the model
    data += model.eval(params=params, x=x)

    # Perform the fit
    result = model.fit(data, params, x=x)

    # Assert that the fit was successful
    assert result.success

    # Assert that the fitted parameters are close to the true values
    assert np.isclose(result.params["center"].value, 0.0)
    assert np.isclose(result.params["temp"].value, 30.0)
    assert np.isclose(result.params["resolution"].value, 0.02)

    # Assert that the fitted curve matches the test data
    assert np.allclose(result.best_fit, data)


def test_fermi_edge_model() -> None:
    # Create a test input DataArray
    x = np.linspace(-0.1, 0.1, 100)
    data = np.zeros_like(x)

    # Create an instance of FermiEdgeModel
    model = models.FermiEdgeModel()

    # Set initial parameter values
    params = model.make_params()
    params["center"].set(value=0.0)
    params["temp"].set(value=30.0)
    params["resolution"].set(value=0.02)
    params["back0"].set(value=0.0)
    params["back1"].set(value=0.0)
    params["dos0"].set(value=1.0)
    params["dos1"].set(value=0.0)

    # Generate test data based on the model
    data += model.eval(params=params, x=x)

    # Perform the fit
    result = model.fit(data, params, x=x)

    # Assert that the fit was successful
    assert result.success

    # Assert that the fitted parameters are close to the true values
    assert np.isclose(result.params["center"].value, 0.0)
    assert np.isclose(result.params["temp"].value, 30.0)
    assert np.isclose(result.params["resolution"].value, 0.02)
    assert np.isclose(result.params["back0"].value, 0.0)
    assert np.isclose(result.params["back1"].value, 0.0)
    assert np.isclose(result.params["dos0"].value, 1.0)
    assert np.isclose(result.params["dos1"].value, 0.0)

    # Assert that the fitted curve matches the test data
    assert np.allclose(result.best_fit, data)


def test_fermi_edge_2d_model() -> None:
    # Create a test input DataArray
    eV = np.linspace(-1.1, 0.1, 100)
    alpha = np.linspace(-10.0, 10.0, 100)
    data = xr.DataArray(
        np.random.default_rng(1).random((100, 100), dtype=np.float64),
        dims=["eV", "alpha"],
        coords={"eV": eV, "alpha": alpha},
    )
    data.attrs["sample_temp"] = 300.0

    # Create an instance of FermiEdge2dModel
    model = models.FermiEdge2dModel(degree=2)

    # Call the guess method
    result = model.guess(data, eV=eV, alpha=alpha)

    # Check if the parameters are set correctly
    assert result["c0"].value == 0.0
    assert result["c1"].value == 0.0
    assert result["c2"].value == 0.0
    assert result["const_bkg"].value > 0.0
    assert result["lin_bkg"].value > 0.0
    assert result["temp"].value == 300.0

    # Call guess with meshgrid
    eV, alpha = np.meshgrid(eV, alpha, indexing="ij")
    result = model.guess(data.values, eV=eV, alpha=alpha)

    # Check if the parameters are set correctly
    assert result["c0"].value == 0.0
    assert result["c1"].value == 0.0
    assert result["c2"].value == 0.0
    assert result["const_bkg"].value > 0.0
    assert result["lin_bkg"].value > 0.0

    # Perform the fit
    fit_res = data.xlm.modelfit(["eV", "alpha"], model=model, guess=True)

    # Try correcting with fit result
    erlab.analysis.gold.correct_with_edge(data, fit_res)


def test_multi_peak_model() -> None:
    # Create test data
    x = np.linspace(-10, 10, 100)
    y = np.zeros_like(x)

    # Create a MultiPeakModel instance (disable convolution to keep fits stable)
    model = models.MultiPeakModel(npeaks=2, fd=True, convolve=False)

    # Set initial parameter values
    params = model.make_params()
    params["p0_center"].set(value=-2.0)
    params["p0_height"].set(value=1.0)
    params["p0_width"].set(value=0.5)
    params["p1_center"].set(value=2.0)
    params["p1_height"].set(value=0.5)
    params["p1_width"].set(value=0.3)
    params["efermi"].set(value=0.0)
    params["temp"].set(value=30.0)
    params["offset"].set(value=0.1)

    # Generate test data based on the model
    y += model.eval(params=params, x=x)

    # Perform the fit (limit iterations to avoid runaway optimization)
    result = model.fit(y, params, x=x, max_nfev=50, calc_covar=False)

    # Assert that the fit was successful
    assert result.success

    # Assert that the fitted parameters are close to the true values
    assert np.isclose(result.params["p0_center"].value, -2.0)
    assert np.isclose(result.params["p0_height"].value, 1.0)
    assert np.isclose(result.params["p0_width"].value, 0.5)
    assert np.isclose(result.params["p1_center"].value, 2.0)
    assert np.isclose(result.params["p1_height"].value, 0.5)
    assert np.isclose(result.params["p1_width"].value, 0.3)
    assert np.isclose(result.params["efermi"].value, 0.0)
    assert np.isclose(result.params["temp"].value, 30.0)
    assert np.isclose(result.params["offset"].value, 0.1)

    # Assert that the fitted curve matches the test data
    assert np.allclose(result.best_fit, y)

    # Assert that the components of the fitted curve can be evaluated
    components = model.eval_components(params=result.params, x=x)
    assert np.allclose(
        components["2Peak_p0"], model.func.eval_peak(0, x, **result.params.valuesdict())
    )
    assert np.allclose(
        components["2Peak_p1"], model.func.eval_peak(1, x, **result.params.valuesdict())
    )
    assert np.allclose(
        components["2Peak_linear"], model.func.eval_bkg(x, **result.params.valuesdict())
    )
    assert np.allclose(
        components["2Peak_fd"], model.func.eval_fd(x, **result.params.valuesdict())
    )

    test_darr = xr.DataArray(y, dims=["x"], coords={"x": x})

    # Cut out the middle of the data
    test_darr = test_darr.where((test_darr.x < -3) | (test_darr.x > 3), drop=True)

    # Evaluate a segmented model on gapped data to exercise segmentation path.
    segmented_model = models.MultiPeakModel(
        npeaks=2, fd=True, segmented=True, oversample=1
    )
    segmented_params = segmented_model.make_params()
    for name, par in params.items():
        if name in segmented_params:
            segmented_params[name].set(value=par.value)
    segmented_params["resolution"].set(value=0.02)
    segmented_model.eval(params=segmented_params, x=test_darr.x.values)

    # Make sure guesses work for different backgrounds
    for background in ["constant", "linear", "polynomial", "none", "shirley"]:
        model = models.MultiPeakModel(npeaks=2, background=background)
        model.guess(y, x=x)


def test_multi_peak_model_shirley_components_and_guess() -> None:
    x = np.linspace(-2, 2, 200)
    model = models.MultiPeakModel(
        npeaks=1, fd=False, background="shirley", convolve=False
    )
    params = model.make_params()
    params["p0_center"].set(value=0.2)
    params["p0_height"].set(value=1.0)
    params["p0_width"].set(value=0.4)
    params["k_step_0"].set(value=0.5)
    params["k_slope"].set(value=0.3)
    params["lin_bkg"].set(value=0.1)
    params["const_bkg"].set(value=0.2)

    y = model.eval(params=params, x=x)
    comps = model.eval_components(params=params, x=x)
    assert "1Peak_baseline" in comps
    assert "1Peak_shirley" in comps
    assert "1Peak_slope" in comps
    assert np.allclose(y, sum(comps.values()))

    guess = model.guess(y, x=x)
    assert np.isfinite(guess["k_step_0"].value)
    assert np.isfinite(guess["const_bkg"].value)


@pytest.mark.parametrize("prefix", ["", "m_", "a"])
@pytest.mark.parametrize(("fd", "convolve"), [(False, False), (True, True)])
@pytest.mark.parametrize(
    "background", ["none", "constant", "linear", "polynomial", "shirley"]
)
def test_multi_peak_component_models(prefix, fd, convolve, background) -> None:
    model = models.MultiPeakModel(
        npeaks=2,
        peak_shapes=["gaussian", "voigt"],
        fd=fd,
        convolve=convolve,
        background=background,
        prefix=prefix,
        name="Custom peaks",
    )
    model.set_param_hint("p0_center", value=-0.3)
    model.set_param_hint("p1_center", value=0.1)
    params = model.make_params()
    x = np.linspace(-1.0, 0.5, 61)
    original = model.eval_components(params=params, x=x)
    component_models = model.components
    assert [component._name for component in component_models] == list(original)
    for component in component_models:
        assert isinstance(component, lmfit.Model)
        assert component.prefix == ""
        component_params = component.make_params()
        assert component_params.keys() == params.keys()
        for name, parameter in params.items():
            assert component_params[name].expr == parameter.expr
            assert component_params[name].min == parameter.min
            assert component_params[name].max == parameter.max
        np.testing.assert_allclose(
            component.eval(params=params, x=x), original[component._name]
        )
        np.testing.assert_allclose(
            component.eval(params=component_params, x=x), original[component._name]
        )


@pytest.mark.parametrize("prefix", ["", "m_", "a"])
@pytest.mark.parametrize(("fd", "convolve"), [(False, False), (True, True)])
@pytest.mark.parametrize("composite", [False, True])
def test_multi_peak_lmfit_uncertainty(prefix, fd, convolve, composite) -> None:
    from scipy.stats import t

    peak_model = models.MultiPeakModel(
        npeaks=1,
        peak_shapes="gaussian",
        fd=fd,
        convolve=convolve,
        background="none",
        prefix=prefix,
    )
    model = peak_model
    if composite:
        model = lmfit.models.ConstantModel(prefix="baseline_") + model
    params = model.make_params()
    for name, value in {
        "p0_height": 2.0,
        "p0_center": -0.1,
        "p0_width": 0.3,
        "temp": 30.0,
        "efermi": 0.0,
        "offset": 0.0,
        "resolution": 0.03,
    }.items():
        if prefix + name in params:
            params[prefix + name].set(value=value)
    for parameter in params.values():
        if parameter.expr is None:
            parameter.set(vary=False)
    params[prefix + "p0_height"].set(vary=True)
    if composite:
        params["baseline_c"].set(value=0.4)
    x = np.linspace(-0.8, 0.6, 101)
    data = model.eval(params=params, x=x) + 0.01 * np.sin(np.arange(x.size))
    result = model.fit(data, params=params, x=x)
    params_before = result.params.valuesdict()
    assert result.errorbars
    dense_x = np.linspace(x.min(), x.max(), 151)
    band = result.eval_uncertainty(sigma=0.95, x=dense_x)
    height = result.params[prefix + "p0_height"]
    scale = t.ppf(0.975, result.nfree) * height.stderr
    full_peak = peak_model.eval(params=result.params, x=dense_x)
    np.testing.assert_allclose(
        band, scale * np.abs(full_peak / height.value), atol=1e-12
    )
    components = result.eval_components(x=dense_x)
    assert result.dely_comps.keys() == components.keys()
    peak_name = (prefix if prefix.endswith("_") else (prefix or "1Peak") + "_") + "p0"
    for name, delta in result.dely_comps.items():
        if name == peak_name:
            np.testing.assert_allclose(
                delta, scale * np.abs(components[name] / height.value)
            )
        else:
            np.testing.assert_allclose(delta, 0.0)
    assert result.params.valuesdict() == params_before
    assert all(np.isfinite(delta).all() for delta in result.dely_comps.values())


def test_multi_peak_components_follow_model_updates() -> None:
    model = models.MultiPeakModel(fd=False, convolve=False)
    assert model.components[0]._name == "1Peak_p0"
    model.prefix = "updated_"
    model.set_param_hint("p0_height", value=2.0)
    components = model.components
    assert components[0]._name == "updated_p0"
    assert components[0].make_params()["updated_p0_height"].value == 2.0


@pytest.mark.parametrize("prefix", ["", "m_", "a"])
def test_multi_peak_component_keyword_overrides(prefix) -> None:
    model = models.MultiPeakModel(
        1,
        fd=True,
        convolve=False,
        background="none",
        prefix=prefix,
        independent_vars=["x", "temp"],
    )
    params = model.make_params(temp=300.0, offset=0.0)
    x = np.linspace(-0.1, 0.1, 41)
    defaults = model.eval_components(params=params, x=x)
    for component in model.components:
        np.testing.assert_allclose(
            component.eval(params=params, x=x), defaults[component._name]
        )
    kwargs = {"x": x, "temp": 30.0, prefix + "p0_height": 2.5}
    components = model.eval_components(params=params, **kwargs)
    for component in model.components:
        np.testing.assert_allclose(
            component.eval(params=params, **kwargs), components[component._name]
        )
    peak, fd = components.values()
    np.testing.assert_allclose(peak * fd, model.eval(params=params, **kwargs))
    np.testing.assert_allclose(
        fd, model.func.eval_fd(x, temp=30.0, efermi=0.0, offset=0.0)
    )


@pytest.mark.parametrize("background", ["none", "shirley"])
def test_multi_peak_lmfit_uncertainty_uses_full_covariance(background) -> None:
    from scipy.stats import t

    model = models.MultiPeakModel(
        2, peak_shapes="gaussian", fd=False, convolve=False, background=background
    )
    params = model.make_params(
        p0_center=-0.1,
        p0_height=2.0,
        p0_width=0.3,
        p1_center=0.1,
        p1_height=1.0,
        p1_width=0.35,
    )
    for name, parameter in params.items():
        if parameter.expr is None:
            parameter.set(vary=name in {"p0_height", "p1_height"})
    if background == "shirley":
        params["k_step_0"].set(value=0.0)
    x = np.linspace(-0.8, 0.8, 101)
    data = model.eval(params=params, x=x) + 0.01 * np.sin(np.arange(x.size))
    result = model.fit(data, params=params, x=x)
    assert result.errorbars
    assert abs(result.covar[0, 1]) > 0.1 * np.sqrt(
        result.covar[0, 0] * result.covar[1, 1]
    )
    dense_x = np.linspace(x.min(), x.max(), 151)
    components = result.eval_components(x=dense_x)
    jacobian = np.array(
        [
            components[f"2Peak_p{i}"] / result.params[f"p{i}_height"].value
            for i in range(2)
        ]
    )
    variance = np.einsum("ij,ix,jx->x", result.covar, jacobian, jacobian)
    scale = t.ppf(0.975, result.nfree)
    uncertainty = result.eval_uncertainty(sigma=0.95, x=dense_x)
    np.testing.assert_allclose(uncertainty, scale * np.sqrt(variance), atol=1e-12)
    assert result.dely_comps.keys() == components.keys()
    for i in range(2):
        np.testing.assert_allclose(
            result.dely_comps[f"2Peak_p{i}"],
            scale * np.sqrt(result.covar[i, i]) * np.abs(jacobian[i]),
            atol=1e-12,
        )
    if background == "shirley":
        for name in ("baseline", "shirley", "slope"):
            np.testing.assert_array_equal(result.dely_comps[f"2Peak_{name}"], 0.0)


@pytest.mark.parametrize("npeaks", [0, 1])
@pytest.mark.parametrize("size", [1, 41])
def test_multi_peak_shirley_zero_components(npeaks, size) -> None:
    model = models.MultiPeakModel(
        npeaks, fd=False, convolve=False, background="shirley"
    )
    params = model.make_params()
    if npeaks:
        params["k_step_0"].set(value=0.0)
    x = np.linspace(-1.0, 1.0, size)
    components = model.eval_components(params=params, x=x)
    assert [component._name for component in model.components] == list(components)
    for name in ("shirley", "slope"):
        np.testing.assert_array_equal(
            components[f"{npeaks}Peak_{name}"], np.zeros(size)
        )
    np.testing.assert_allclose(sum(components.values()), model.eval(params=params, x=x))


def test_multi_peak_model_voigt() -> None:
    x = np.linspace(-4, 4, 200)
    model = models.MultiPeakModel(
        npeaks=1, peak_shapes="voigt", fd=False, background="none", convolve=False
    )
    params = model.make_params()
    params["p0_center"].set(value=0.5)
    params["p0_sigma"].set(value=0.25)
    params["p0_gamma"].set(value=0.15)
    params["p0_amplitude"].set(value=1.2)

    expected = voigt(x, 0.5, 0.25, 0.15, 1.2)
    assert np.allclose(model.eval(params=params, x=x), expected)

    guess = model.guess(expected, x=x)
    assert guess["p0_gamma"].value > 0.0
    assert guess["p0_amplitude"].value > 0.0

    assert model.param_hints["p0_width"]["expr"] is not None
    assert model.param_hints["p0_height"]["expr"] is not None


def test_multi_peak_model_guess_finds_peaks() -> None:
    x = np.linspace(-5, 5, 401)
    model = models.MultiPeakModel(npeaks=2, fd=False, background="linear")
    params = model.make_params()
    params["p0_center"].set(value=-1.5)
    params["p0_height"].set(value=1.2)
    params["p0_width"].set(value=0.7)
    params["p1_center"].set(value=2.2)
    params["p1_height"].set(value=0.8)
    params["p1_width"].set(value=1.1)
    params["const_bkg"].set(value=0.2)
    params["lin_bkg"].set(value=0.05)

    y = model.eval(params=params, x=x)
    order = np.random.default_rng(0).permutation(x.size)
    guess = model.guess(y[order], x=x[order])

    centers = np.sort([guess["p0_center"].value, guess["p1_center"].value])
    expected = np.sort([params["p0_center"].value, params["p1_center"].value])
    assert np.allclose(centers, expected, atol=0.25)
    assert guess["p0_height"].value > 0.0
    assert guess["p1_height"].value > 0.0
    assert guess["p0_width"].value > 0.0
    assert guess["p1_width"].value > 0.0


def test_multi_peak_model_guess_fallback_even_spacing() -> None:
    x = np.linspace(-4, 4, 200)
    y = np.full_like(x, 0.2)
    model = models.MultiPeakModel(npeaks=3, fd=False, background="none")

    guess = model.guess(y, x=x)
    centers = np.sort(
        [guess["p0_center"].value, guess["p1_center"].value, guess["p2_center"].value]
    )
    expected = np.linspace(x.min(), x.max(), 5)[1:-1]
    step = x[1] - x[0]

    assert np.allclose(centers, expected, atol=step)
    widths = np.array(
        [guess["p0_width"].value, guess["p1_width"].value, guess["p2_width"].value]
    )
    xrange = x.max() - x.min()
    assert np.all(widths >= 0.1 * xrange)
    assert np.all(widths <= xrange)


def test_multi_peak_model_with_prefix() -> None:
    x = np.linspace(-1.0, 1.0, 51)
    model = models.MultiPeakModel(
        npeaks=1, fd=False, background="none", convolve=False, prefix="m_"
    )
    params = model.make_params()
    assert "m_p0_center" in params
    assert "m_p0_height" in params
    assert "m_p0_width" in params

    params["m_p0_center"].set(value=0.0)
    params["m_p0_height"].set(value=1.0)
    params["m_p0_width"].set(value=0.4)

    y = model.eval(params=params, x=x)
    comps = model.eval_components(params=params, x=x)
    fargs = model.make_funcargs(params, {"x": x})
    fargs_no_x = {k: v for k, v in fargs.items() if k != "x"}

    expected_key = f"{model._prefix}p0"
    assert expected_key in comps
    assert np.allclose(comps[expected_key], model.func.eval_peak(0, x, **fargs_no_x))
    assert np.allclose(y, sum(comps.values()))


def test_polynomial_model() -> None:
    # Create test data
    x = np.linspace(-10, 10, 100)
    y = np.polyval([3, -2, 1], x)

    # Create an instance of PolynomialModel
    model = models.PolynomialModel(degree=2)

    # Set initial parameter values
    # Guess with only y
    params = model.guess(y)
    # Guess with x and y
    params = model.guess(y, x=x)
    params["c0"].set(value=1.0)
    params["c1"].set(value=-2.0)
    params["c2"].set(value=3.0)

    # Perform the fit
    result = model.fit(y, params, x=x)

    # Assert that the fit was successful
    assert result.success

    # Assert that the fitted parameters are close to the true values
    assert np.isclose(result.params["c0"].value, 1.0)
    assert np.isclose(result.params["c1"].value, -2.0)
    assert np.isclose(result.params["c2"].value, 3.0)

    # Assert that the fitted curve matches the test data
    assert np.allclose(result.best_fit, y)


def test_step_edge_model() -> None:
    # Create a test input DataArray
    x = np.linspace(-10, 10, 100)
    data = np.zeros_like(x)

    # Create an instance of StepEdgeModel
    model = models.StepEdgeModel()

    # Set initial parameter values
    params = model.make_params()
    params["center"].set(value=0.0)
    params["back0"].set(value=0.0)
    params["back1"].set(value=0.0)
    params["dos0"].set(value=1.0)
    params["dos1"].set(value=0.01)
    params["sigma"].set(value=0.5)

    # Generate test data based on the model
    data += model.eval(params=params, x=x)

    # Perform the fit
    result = model.fit(data, params, x=x)

    # Assert that the fit was successful
    assert result.success

    # Assert that the fitted parameters are close to the true values
    assert np.isclose(result.params["center"].value, 0.0)
    assert np.isclose(result.params["back0"].value, 0.0)
    assert np.isclose(result.params["back1"].value, 0.0)
    assert np.isclose(result.params["dos0"].value, 1.0)
    assert np.isclose(result.params["dos1"].value, 0.01)
    assert np.isclose(result.params["sigma"].value, 0.5)

    # Assert that the fitted curve matches the test data
    assert np.allclose(result.best_fit, data)


def test_get_edges() -> None:
    # Basic test with fraction=0.2
    x = np.arange(20)
    y = np.arange(20)
    edges_x, edges_y = models._get_edges(x, y, fraction=0.2)
    num = max(5, round(len(x) * 0.2))
    assert len(edges_x) == 2 * num
    assert np.array_equal(edges_x, np.r_[x[:num], x[-num:]])
    assert np.array_equal(edges_y, np.r_[y[:num], y[-num:]])

    # fraction = 0 (should select 5 elements from each edge)
    edges_x, edges_y = models._get_edges(x, y, fraction=0)
    assert len(edges_x) == 10
    # fraction = 0.5 (should select half from each edge)
    edges_x, edges_y = models._get_edges(x, y, fraction=0.5)
    assert len(edges_x) == 20

    # Test errors
    with pytest.raises(ValueError, match=r"Fraction must be between 0 and 0.5"):
        models._get_edges(x, y, fraction=-0.1)
    with pytest.raises(ValueError, match=r"Fraction must be between 0 and 0.5"):
        models._get_edges(x, y, fraction=0.6)

    # Test with mismatched lengths
    x = np.arange(20)
    y = np.arange(19)
    with pytest.raises(ValueError, match=r"x and y must have the same length"):
        models._get_edges(x, y)

    # Test with arrays that are too short
    x = np.arange(9)
    y = np.arange(9)

    with pytest.raises(ValueError, match=r"Array must have at least 10 elements"):
        models._get_edges(x, y)


def test_tll_model_params_and_eval() -> None:
    x = np.linspace(-0.05, 0.05, 11)
    model = models.TLLModel()
    params = model.make_params()
    assert params["amp"].min == 0.0
    assert params["alpha"].min == 0.0
    assert params["temp"].min == 0.0
    assert params["resolution"].min == 0.0

    params["amp"].set(value=1.2)
    params["center"].set(value=0.01)
    params["alpha"].set(value=0.2)
    params["temp"].set(value=20.0)
    params["resolution"].set(value=0.02)
    params["const_bkg"].set(value=0.03)

    result = model.eval(params=params, x=x)
    expected = tll(x, **params.valuesdict())
    assert np.allclose(result, expected)


def test_symmetrized_gap_model_params_and_eval() -> None:
    x = np.linspace(-0.1, 0.1, 21)
    model = models.SymmetrizedGapModel()
    params = model.make_params()
    assert params["amp"].min == 0.0
    assert params["gamma1"].min == 0.0
    assert params["gamma0"].value == 0.0
    assert params["gamma0"].vary is False
    assert params["delta"].min == 0.0
    assert params["resolution"].min == 0.0

    params["amp"].set(value=1.1)
    params["gamma1"].set(value=0.02)
    params["gamma0"].set(value=0.0, vary=False)
    params["delta"].set(value=0.03)
    params["lin_bkg"].set(value=0.2)
    params["const_bkg"].set(value=0.1)
    params["resolution"].set(value=0.01)

    result = model.eval(params=params, x=x)
    expected = sc_spectral_function(x, **params.valuesdict())
    assert np.allclose(result, expected)


@pytest.mark.parametrize("prefix", ["", "main_"])
@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize(
    ("model_class", "model_kwargs", "parameter_values", "varying_parameter"),
    [
        pytest.param(models.FermiEdgeModel, {}, {}, "dos0", id="fermi-edge"),
        pytest.param(models.FermiDiracModel, {}, {}, "center", id="fermi-dirac"),
        pytest.param(models.StepEdgeModel, {}, {}, "dos0", id="step-edge"),
        pytest.param(
            models.PolynomialModel,
            {"degree": 2},
            {"c0": 1.0, "c1": 0.5, "c2": 0.3},
            "c0",
            id="polynomial",
        ),
        pytest.param(
            models.FermiEdge2dModel,
            {"degree": 2},
            {"c0": 0.0, "c1": 0.001, "c2": 0.0},
            "const_bkg",
            id="fermi-edge-2d",
        ),
        pytest.param(models.BCSGapModel, {}, {}, "a", id="bcs-gap"),
        pytest.param(models.DynesModel, {}, {}, "n0", id="dynes"),
        pytest.param(models.TLLModel, {}, {}, "amp", id="tll"),
        pytest.param(
            models.SymmetrizedGapModel,
            {},
            {"gamma1": 0.005, "delta": 0.01},
            "amp",
            id="symmetrized-gap",
        ),
    ],
)
def test_single_component_model_uncertainty(
    model_class, model_kwargs, parameter_values, varying_parameter, prefix, composite
) -> None:
    model = model_class(prefix=prefix, **model_kwargs)
    params = model.make_params(**parameter_values)
    if composite:
        secondary = model_class(prefix="secondary_", **model_kwargs)
        params.update(secondary.make_params(**parameter_values))
        model += secondary

    for parameter in params.values():
        parameter.vary = False
    params[f"{prefix}{varying_parameter}"].vary = True

    if model_class is models.FermiEdge2dModel:
        independent = {
            "eV": np.linspace(-0.1, 0.1, 81),
            "alpha": np.linspace(-3.0, 3.0, 9),
        }
    else:
        independent = {
            "x": (
                np.linspace(1.0, 90.0, 101)
                if model_class is models.BCSGapModel
                else np.linspace(-0.1, 0.1, 101)
            )
        }

    expected = model.eval(params=params, **independent)
    noise = np.random.default_rng(0).normal(
        scale=max(np.ptp(expected), 1.0) * 1e-4, size=expected.shape
    )
    result = model.fit(expected + noise, params=params, **independent)
    assert result.success
    assert result.covar is not None

    components = result.eval_components(**independent)
    labels = [component.prefix or component._name for component in model.components]
    assert len(labels) == (2 if composite else 1)
    assert labels == list(components)
    for component, label in zip(model.components, labels, strict=True):
        np.testing.assert_allclose(
            component.eval(params=result.params, **independent), components[label]
        )

    uncertainty = result.eval_uncertainty(**independent)
    assert uncertainty.shape == expected.shape
    assert np.all(np.isfinite(uncertainty))
    assert np.any(uncertainty > 0.0)
    assert set(result.dely_comps) == set(labels)
    np.testing.assert_allclose(result.dely_comps[labels[0]], uncertainty, atol=1e-12)
    if composite:
        np.testing.assert_allclose(result.dely_comps[labels[1]], 0.0)
