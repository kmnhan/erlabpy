import numpy as np
import pytest
import xarray as xr
from erlab.analysis.fit.functions.dynamic import (
    FermiEdge2dFunction,
    MultiPeakFunction,
    PolynomialFunction,
)
from erlab.analysis.fit.functions.general import gaussian_wh, lorentzian_wh

RAND_STATE = np.random.RandomState(1)


def test_poly_func_call():
    # Test case 1: Evaluate polynomial function with numpy array input
    x = np.arange(5, dtype=np.float64)
    coeffs = RAND_STATE.randn(3)
    expected_result = np.polyval(np.asarray(list(reversed(coeffs))), x)
    assert np.allclose(
        PolynomialFunction(degree=2)(x, *coeffs),
        expected_result,
    )

    # Test case 2: Evaluate polynomial function with xarray input
    x = xr.DataArray(np.arange(5, dtype=np.float64), dims="x")
    coeffs = RAND_STATE.randn(3)
    expected_result = np.polyval(np.asarray(list(reversed(coeffs))), x)
    result = PolynomialFunction(degree=2)(x, *coeffs)
    assert isinstance(result, xr.DataArray)
    assert np.allclose(result, expected_result)

    # Test case 3: Evaluate polynomial function with parameters
    x = np.arange(5, dtype=np.float64)
    coeffs = RAND_STATE.randn(3)
    expected_result = np.polyval(np.asarray(list(reversed(coeffs))), x)
    params = dict(zip([f"c{i}" for i in range(3)], coeffs, strict=True))
    result = PolynomialFunction(degree=2)(x, **params)
    assert np.allclose(result, expected_result)


def test_multi_peak_function_call():
    # Test case 1: Evaluate multi-peak function with numpy array input
    x = np.linspace(-5, 5, 20, dtype=np.float64)
    npeaks = 2
    peak_shapes = ["lorentzian", "gaussian"]
    fd = False
    convolve = False
    params = {
        "p0_center": 0.0,
        "p0_height": 1.0,
        "p0_width": 0.5,
        "p1_center": 2.0,
        "p1_height": 0.5,
        "p1_width": 0.2,
        "lin_bkg": 0.1,
        "const_bkg": 0.2,
    }
    expected_result = (
        lorentzian_wh(x, params["p0_center"], params["p0_width"], params["p0_height"])
        + gaussian_wh(x, params["p1_center"], params["p1_width"], params["p1_height"])
        + params["lin_bkg"] * x
        + params["const_bkg"]
    )
    assert np.allclose(
        MultiPeakFunction(npeaks, peak_shapes, fd=fd, convolve=convolve)(x, **params),
        expected_result,
    )
    # Test case 2: With different peak shape signature
    assert np.allclose(
        MultiPeakFunction(npeaks, "l g", fd=fd, convolve=convolve)(x, **params),
        expected_result,
    )

    # Test case 3: Evaluate multi-peak function with xarray input
    x = xr.DataArray(np.linspace(-5, 5, 20, dtype=np.float64), dims="x")
    result_xr = MultiPeakFunction(npeaks, peak_shapes, fd=fd, convolve=convolve)(
        x, **params
    )
    assert isinstance(result_xr, xr.DataArray)
    assert np.allclose(result_xr, expected_result)

    # Test case 4: Evaluate multi-peak function with different peak shape signatures
    x = np.linspace(-5, 5, 20, dtype=np.float64)
    npeaks = 3
    peak_shapes = ["gaussian", "lorentzian", "gaussian"]
    fd = True
    convolve = True
    params = {
        "p0_center": 1.0,
        "p0_height": 0.5,
        "p0_width": 0.2,
        "p1_center": 2.0,
        "p1_height": 1.0,
        "p1_width": 0.5,
        "p2_center": 3.0,
        "p2_height": 0.8,
        "p2_width": 0.3,
        "lin_bkg": 0.2,
        "const_bkg": 0.1,
        "efermi": 0.0,
        "temp": 30,
        "offset": 0.5,
        "resolution": 0.1,
    }

    expected_result = MultiPeakFunction(npeaks, "lorentzian", fd=fd, convolve=convolve)(
        x, **params
    )

    for peak_shapes in ("l l l", ["lorentzian"] * 3):
        assert np.allclose(
            MultiPeakFunction(npeaks, peak_shapes, fd=fd, convolve=convolve)(
                x, **params
            ),
            expected_result,
        )
    # Test case 5: Evaluate multi-peak function with different background shapes
    x = np.linspace(-5, 5, 20, dtype=np.float64)
    npeaks = 1
    peak_shapes = ["lorentzian"]
    fd = False
    convolve = False
    degree = 2

    params = {
        "p0_center": 0.0,
        "p0_height": 1.0,
        "p0_width": 0.5,
    }

    for background in ("constant", "linear", "polynomial", "none"):
        expected_result = lorentzian_wh(
            x, params["p0_center"], params["p0_width"], params["p0_height"]
        )

        if background == "constant":
            params["const_bkg"] = 0.1
            expected_bkg = params["const_bkg"]

        elif background == "linear":
            params["const_bkg"] = 0.1
            params["lin_bkg"] = 0.2
            expected_bkg = params["lin_bkg"] * x + params["const_bkg"]

        elif background == "polynomial":
            for d in range(degree + 1):
                params[f"c{d}"] = 0.1 * degree

            coeffs = tuple(params[f"c{d}"] for d in range(degree + 1))
            expected_bkg = np.polynomial.polynomial.polyval(x, coeffs)
        else:
            expected_bkg = 0.0

        assert np.allclose(
            MultiPeakFunction(
                npeaks,
                peak_shapes,
                background=background,
                degree=degree,
                fd=fd,
                convolve=convolve,
            )(x, **params),
            expected_result + expected_bkg,
        )


@pytest.mark.filterwarnings("ignore:overflow encountered in exp.*:RuntimeWarning")
def test_fermi_edge_2d_function_call():
    # Difficult to test function values, just check if equal for numpy and xarray
    eV = np.linspace(-5, 5, 25, dtype=np.float64)
    alpha = RAND_STATE.randn(20)
    alpha.sort()
    params = {
        "c0": 1.0,
        "c1": 0.5,
        "lin_bkg": 0.1,
        "const_bkg": 0.2,
        "offset": 0.0,
        "temp": 30.0,
        "resolution": 0.02,
    }
    out_np = FermiEdge2dFunction()(eV, alpha, **params)

    eV = xr.DataArray(eV, dims="eV")
    alpha = xr.DataArray(alpha, dims="alpha")
    out_xr = FermiEdge2dFunction()(eV, alpha, **params)

    assert isinstance(out_xr, xr.DataArray)
    assert np.allclose(out_np.reshape((25, 20)), out_xr)
