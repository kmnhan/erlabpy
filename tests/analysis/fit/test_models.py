import erlab.analysis.fit.models as models
import numpy as np
import xarray as xr


def test_fermi_edge_model():
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


def test_fermi_edge_2d_model():
    # Create a test input DataArray
    eV = np.linspace(-1.1, 0.1, 100)
    alpha = np.linspace(-10.0, 10.0, 100)
    data = xr.DataArray(
        np.random.default_rng(1).random((100, 100), dtype=np.float64),
        dims=["eV", "alpha"],
        coords={"eV": eV, "alpha": alpha},
    )
    data.attrs["temp_sample"] = 300.0

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


def test_multi_peak_model():
    # Create test data
    x = np.linspace(-10, 10, 100)
    y = np.zeros_like(x)

    # Create a MultiPeakModel instance
    model = models.MultiPeakModel(npeaks=2)

    # Set initial parameter values
    params = model.make_params()
    params["p0_center"].set(value=-2.0)
    params["p0_height"].set(value=1.0)
    params["p0_width"].set(value=0.5)
    params["p1_center"].set(value=2.0)
    params["p1_height"].set(value=0.5)
    params["p1_width"].set(value=0.3)

    # Generate test data based on the model
    y += model.eval(params=params, x=x)

    # Perform the fit
    result = model.fit(y, params, x=x)

    # Assert that the fit was successful
    assert result.success

    # Assert that the fitted parameters are close to the true values
    assert np.isclose(result.params["p0_center"].value, -2.0)
    assert np.isclose(result.params["p0_height"].value, 1.0)
    assert np.isclose(result.params["p0_width"].value, 0.5)
    assert np.isclose(result.params["p1_center"].value, 2.0)
    assert np.isclose(result.params["p1_height"].value, 0.5)
    assert np.isclose(result.params["p1_width"].value, 0.3)

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
        components["2Peak_bkg"], model.func.eval_bkg(x, **result.params.valuesdict())
    )

    # Make sure guesses work for different backgrounds
    for background in ["constant", "linear", "polynomial", "none"]:
        model = models.MultiPeakModel(npeaks=2, background=background)
        params = model.guess(y, x=x)
        model.fit(y, params, x=x)


def test_polynomial_model():
    # Create test data
    x = np.linspace(-10, 10, 100)
    y = np.polyval([3, -2, 1], x)

    # Create an instance of PolynomialModel
    model = models.PolynomialModel(degree=2)

    # Set initial parameter values
    params = model.make_params()
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


def test_step_edge_model():
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
