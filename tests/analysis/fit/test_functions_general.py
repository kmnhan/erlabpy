import numpy as np
import pytest
import scipy.special

from erlab.analysis.fit.functions.general import (
    _gen_kernel,
    do_convolve,
    fermi_dirac,
    fermi_dirac_broad,
    fermi_dirac_linbkg,
    fermi_dirac_linbkg_broad,
    gaussian_wh,
    lorentzian_wh,
    step_broad,
    step_linbkg_broad,
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


def test_do_convolve() -> None:
    # Test case 1: Convolve a quadratic function with a Gaussian kernel
    x = np.linspace(1, 10, 5)

    def testfunc(x):
        return x**2

    resolution = 3.0
    expected = np.array(
        [2.54995914, 12.11245914, 31.79995914, 61.61245914, 101.54995914]
    )
    result = do_convolve(x, testfunc, resolution)
    assert np.allclose(result, expected)

    # Test case 2: Convolve a sine function with a Gaussian kernel
    x = np.linspace(0, 2 * np.pi, 5)

    def testfunc(x):
        return np.sin(x)

    resolution = 5
    expected = np.array(
        [
            -2.60089778e-16,
            1.04956310e-01,
            1.85198936e-16,
            -1.04956310e-01,
            -2.01286051e-16,
        ]
    )
    result = do_convolve(x, testfunc, resolution)
    assert np.allclose(result, expected)


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
