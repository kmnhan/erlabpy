import numpy as np
import pytest
import scipy.signal
import scipy.special

from erlab.analysis.fit.functions.general import (
    _gen_kernel,
    _sc_spectral_function_bare,
    _tll_bare,
    active_shirley,
    do_convolve,
    fermi_dirac,
    fermi_dirac_broad,
    fermi_dirac_linbkg,
    fermi_dirac_linbkg_broad,
    gaussian_wh,
    lorentzian_wh,
    right_integral_trapz,
    sc_spectral_function,
    step_broad,
    step_linbkg_broad,
    tll,
)

KB_EV = 8.617333262145179e-5


def test_gen_kernel() -> None:
    # Define test input values
    x = np.array([0, 1, 2, 3, 4])
    resolution = 2.0
    pad = 5

    # Call the function to generate the kernel
    extended, gauss = _gen_kernel(x, resolution, pad)

    # Define the expected results
    expected_extended = np.arange(-4, 9, 1, dtype=np.float64)
    expected_gauss = np.array(
        [
            7.16733764e-06,
            9.17419217e-04,
            2.93574150e-02,
            2.34859320e-01,
            4.69718639e-01,
            2.34859320e-01,
            2.93574150e-02,
            9.17419217e-04,
            7.16733764e-06,
        ]
    )

    # Check if the generated extended and gauss arrays match the expected results
    assert np.allclose(extended, expected_extended)
    assert np.allclose(gauss, expected_gauss)


def test_do_convolve_matches_kernel() -> None:
    x = np.linspace(1, 10, 5)

    def testfunc(x):
        return x**2

    resolution = 3.0
    xn, g = _gen_kernel(x, resolution)
    expected = scipy.signal.convolve(testfunc(xn), g, mode="valid")
    result = do_convolve(x, testfunc, resolution, oversample=1)
    assert np.allclose(result, expected)

    x = np.linspace(0, 2 * np.pi, 5)

    def testfunc(x):
        return np.sin(x)

    resolution = 5
    xn, g = _gen_kernel(x, resolution)
    expected = scipy.signal.convolve(testfunc(xn), g, mode="valid")
    result = do_convolve(x, testfunc, resolution, oversample=1)
    assert np.allclose(result, expected)


def test_do_convolve_oversample_and_zero_resolution() -> None:
    x = np.linspace(-1, 1, 9)

    def testfunc(x):
        return x**3

    xn_fine = np.linspace(x[0], x[-1], (x.size - 1) * 3 + 1)
    xn_kernel, g = _gen_kernel(xn_fine, 0.5)
    expected = scipy.signal.convolve(testfunc(xn_kernel), g, mode="valid")[::3]
    result = do_convolve(x, testfunc, 0.5)
    assert np.allclose(result, expected)

    result = do_convolve(x, testfunc, 0.0)
    assert np.allclose(result, testfunc(x))


def test_do_convolve_descending_axis_matches_ascending() -> None:
    x = np.linspace(-1.0, 1.0, 401)
    xd = x[::-1]
    params = {"center": 0.2, "temp": 30.0}

    asc = do_convolve(x, fermi_dirac, resolution=0.05, **params)
    desc = do_convolve(xd, fermi_dirac, resolution=0.05, **params)

    np.testing.assert_allclose(desc[::-1], asc, atol=1e-12, rtol=1e-9)


def test_gaussian_wh() -> None:
    x = np.linspace(0, 10, 100)
    center = 5.0
    width = 2.0
    height = 1.0
    expected_result = height * np.exp(
        -16 * np.log(2) * (1.0 * x - center) ** 2 / max(1e-10, width**2)
    )
    assert np.allclose(gaussian_wh(x, center, width, height), expected_result)


def test_lorentzian_wh() -> None:
    x = np.linspace(0, 10, 100)
    center = 5.0
    width = 2.0
    height = 1.0
    expected_result = height / (1 + 4 * ((1.0 * x - center) / max(1.0e-15, width)) ** 2)
    assert np.allclose(lorentzian_wh(x, center, width, height), expected_result)


@pytest.mark.filterwarnings("ignore:overflow encountered in exp.*:RuntimeWarning")
def test_fermi_dirac() -> None:
    x = np.linspace(0, 10, 100)
    center = 5.0
    temp = 2.0
    expected_result = 1 / (1 + np.exp((1.0 * x - center) / max(1.0e-15, temp * KB_EV)))
    assert np.allclose(fermi_dirac(x, center, temp), expected_result)


@pytest.mark.filterwarnings("ignore:overflow encountered in exp.*:RuntimeWarning")
def test_fermi_dirac_linbkg() -> None:
    x = np.linspace(0, 10, 100)
    center = 5.0
    temp = 2.0
    back0 = 1.0
    back1 = 0.5
    dos0 = 0.8
    dos1 = 0.3
    expected_result = (back0 + back1 * x) + (dos0 - back0 + (dos1 - back1) * x) / (
        1 + np.exp((1.0 * x - center) / max(1.0e-15, temp * KB_EV))
    )
    assert np.allclose(
        fermi_dirac_linbkg(x, center, temp, back0, back1, dos0, dos1), expected_result
    )


def test_fermi_dirac_broad() -> None:
    x = np.linspace(0, 10, 100)
    center = 5.0
    temp = 300.0
    resolution = 0.1
    expected_result = do_convolve(
        x,
        fermi_dirac,
        resolution=resolution,
        center=center,
        temp=temp,
    )
    assert np.allclose(
        fermi_dirac_broad(x, center, temp, resolution),
        expected_result,
    )


def test_fermi_dirac_broad_descending_axis_matches_ascending() -> None:
    x = np.linspace(-1.0, 1.0, 401)
    xd = x[::-1]
    center = 0.2
    temp = 30.0
    resolution = 0.05

    asc = fermi_dirac_broad(x, center, temp, resolution)
    desc = fermi_dirac_broad(xd, center, temp, resolution)

    np.testing.assert_allclose(desc[::-1], asc, atol=1e-12, rtol=1e-9)


def test_fermi_dirac_linbkg_broad() -> None:
    x = np.linspace(0, 10, 100)
    center = 5.0
    temp = 300.0
    resolution = 0.1
    back0 = 0.1
    back1 = 0.2
    dos0 = 0.3
    dos1 = 0.4
    expected_result = do_convolve(
        x,
        fermi_dirac_linbkg,
        resolution=resolution,
        center=center,
        temp=temp,
        back0=back0,
        back1=back1,
        dos0=dos0,
        dos1=dos1,
    )
    assert np.allclose(
        fermi_dirac_linbkg_broad(x, center, temp, resolution, back0, back1, dos0, dos1),
        expected_result,
    )


def test_step_broad() -> None:
    x = np.linspace(-10, 10, 100)
    center = 0.0
    sigma = 1.0
    amplitude = 1.0
    expected_result = (
        amplitude
        * 0.5
        * scipy.special.erfc((1.0 * x - center) / max(1.0e-15, np.sqrt(2) * sigma))
    )
    assert np.allclose(step_broad(x, center, sigma, amplitude), expected_result)


def test_step_linbkg_broad() -> None:
    x = np.linspace(0, 10, 100)
    center = 5.0
    sigma = 2.0
    back0 = 1.0
    back1 = 0.5
    dos0 = 2.0
    dos1 = 1.0
    expected_result = (back0 + back1 * x) + (dos0 - back0 + (dos1 - back1) * x) * (
        step_broad(x, center, sigma, 1.0)
    )
    assert np.allclose(
        step_linbkg_broad(x, center, sigma, back0, back1, dos0, dos1), expected_result
    )


def test_tll_resolution_zero_matches_bare() -> None:
    x = np.linspace(-0.1, 0.1, 21)
    params = {"amp": 1.2, "center": 0.01, "alpha": 0.2, "temp": 25.0}
    expected = _tll_bare(x, **params) + 0.03
    result = tll(x, resolution=0.0, const_bkg=0.03, **params)
    assert np.allclose(result, expected)


def test_sc_spectral_function_resolution_zero_matches_bare() -> None:
    x = np.linspace(-0.1, 0.1, 21)
    params = {
        "amp": 1.1,
        "gamma1": 0.02,
        "gamma0": 0.01,
        "delta": 0.03,
        "lin_bkg": 0.2,
        "const_bkg": 0.1,
    }
    expected = _sc_spectral_function_bare(x, **params)
    result = sc_spectral_function(x, resolution=0.0, **params)
    assert np.allclose(result, expected)


def test_right_integral_trapz_constant_increasing_and_decreasing() -> None:
    x_inc = np.linspace(-2.0, 2.0, 9)
    y = np.ones_like(x_inc)
    expected_inc = x_inc[-1] - x_inc
    result_inc = right_integral_trapz(x_inc, y)
    assert np.allclose(result_inc, expected_inc)

    x_dec = x_inc[::-1]
    result_dec = right_integral_trapz(x_dec, y[::-1])
    expected_dec = x_dec[0] - x_dec
    assert np.allclose(result_dec, expected_dec)


def test_right_integral_trapz_errors() -> None:
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.ones_like(x)
    with pytest.raises(ValueError, match="repeated values"):
        right_integral_trapz(x, y)

    x = np.array([0.0, 1.0, 0.5, 2.0])
    with pytest.raises(ValueError, match="monotonic"):
        right_integral_trapz(x, y)

    x2d = np.array([[0.0, 1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="1D arrays"):
        right_integral_trapz(x2d, y)

    with pytest.raises(ValueError, match="same length"):
        right_integral_trapz(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0]))


def test_right_integral_trapz_short_input_returns_zero() -> None:
    x = np.array([0.0])
    y = np.array([2.0])
    result = right_integral_trapz(x, y)
    assert np.allclose(result, np.zeros_like(y))


def test_active_shirley_components() -> None:
    x = np.linspace(0.0, 4.0, 5)
    peak = np.ones_like(x)
    out = active_shirley(
        x,
        peaks=[peak],
        k_steps=[0.5],
        k_slope=0.25,
        lin_bkg=0.1,
        const_bkg=0.2,
    )
    assert set(out) == {"baseline", "shirley", "slope"}
    expected_baseline = 0.2 + 0.1 * x
    assert np.allclose(out["baseline"], expected_baseline)
    assert np.all(out["shirley"] >= 0.0)
    assert np.all(out["slope"] >= 0.0)

    out_no_peaks = active_shirley(
        x, peaks=[], k_steps=[], k_slope=0.0, lin_bkg=0.2, const_bkg=0.1
    )
    assert set(out_no_peaks) == {"baseline"}


def test_active_shirley_invalid_inputs_and_zero_steps() -> None:
    x = np.linspace(0.0, 1.0, 5)
    peak = np.ones_like(x)

    with pytest.raises(ValueError, match="x must be 1D"):
        active_shirley(x.reshape(1, -1), peaks=[peak], k_steps=[0.1])

    x_short = np.array([0.0])
    out_short = active_shirley(
        x_short, peaks=[np.array([1.0])], k_steps=[0.1], const_bkg=0.3
    )
    assert set(out_short) == {"baseline"}
    assert np.allclose(out_short["baseline"], np.array([0.3]))

    with pytest.raises(ValueError, match="k_steps"):
        active_shirley(x, peaks=[peak, peak], k_steps=[0.1])

    out_zero = active_shirley(
        x, peaks=[peak, peak], k_steps=[0.0, 0.2], k_slope=0.0, lin_bkg=0.0
    )
    assert "shirley" in out_zero
