import numpy as np
from erlab.analysis.image import _ndsavgol_coeffs, ndsavgol
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from scipy.signal import savgol_coeffs


def compare_coeffs_to_scipy(window_length, order, deriv):
    # For the given window_length and order, compare the results to scipy
    h1 = _ndsavgol_coeffs((window_length,), order, deriv_idx=deriv, delta=(1.0,))
    h2 = savgol_coeffs(window_length, order, deriv=deriv, use="dot")
    assert_allclose(
        h1,
        h2,
        atol=1e-10,
        err_msg=f"[window_length = {window_length}, order = {order}, deriv={deriv}]",
    )


def test_ndsavgol_coeffs_compare_scipy():
    # Compare _ndsavgol_coeffs() to savgol_coeffs().
    for window_length in range(1, 10, 2):
        for order in range(window_length):
            for deriv in range(order):
                compare_coeffs_to_scipy(window_length, order, deriv)


def test_ndsavgol_trivial():
    # Adapted from scipy tests
    x = np.array([1.0])
    y = ndsavgol(x, 1, 0)
    assert_equal(y, [1.0])

    # Input is a single value. With a window length of 3 and polyorder 1,
    # the value in y is from the straight-line fit of (-1,0), (0,3) and
    # (1, 0) at 0. This is just the average of the three values, hence 1.0.
    x = np.array([3.0])
    y = ndsavgol(x, 3, 1, mode="constant")
    assert_almost_equal(y, [1.0], decimal=15)

    x = np.array([3.0])
    y = ndsavgol(x, 3, 1, mode="nearest")
    assert_almost_equal(y, [3.0], decimal=15)

    x = np.array([1.0] * 3)
    y = ndsavgol(x, 3, 1, mode="wrap")
    assert_almost_equal(y, [1.0, 1.0, 1.0], decimal=15)


def test_ndsavgol_basic():
    # Some basic test cases for savgol_filter().
    x = np.array([1.0, 2.0, 1.0])
    y = ndsavgol(x, 3, 1, mode="constant")
    assert_allclose(y, [1.0, 4.0 / 3, 1.0])

    y = ndsavgol(x, 3, 1, mode="mirror")
    assert_allclose(y, [5.0 / 3, 4.0 / 3, 5.0 / 3])

    y = ndsavgol(x, 3, 1, mode="wrap")
    assert_allclose(y, [4.0 / 3, 4.0 / 3, 4.0 / 3])


def test_ndsavgol_2d():
    # Test 2D savgol_filter() with a 2D Gaussian.
    x, y = np.meshgrid(np.linspace(-3, 3, 5), np.linspace(-3, 3, 5))
    z = np.exp(-(x**2 + y**2))

    # Test derivative argument as integer
    for order in range(4):
        for deriv in range(order - 1):
            assert_equal(
                ndsavgol(z, (3, 3), order, deriv=deriv),
                ndsavgol(z, (3, 3), order, deriv=(deriv, deriv)),
            )

    # Test smoothing
    expected = np.array(
        [
            [0.00247156, 0.01295438, 0.01418799, 0.01295438, 0.00247156],
            [0.01295438, 0.06789891, 0.07436473, 0.06789891, 0.01295438],
            [0.01418799, 0.07436473, 0.08144627, 0.07436473, 0.01418799],
            [0.01295438, 0.06789891, 0.07436473, 0.06789891, 0.01295438],
            [0.00247156, 0.01295438, 0.01418799, 0.01295438, 0.00247156],
        ]
    )
    assert_allclose(ndsavgol(z, (3, 3), 1, deriv=0), expected, atol=1e-5)

    # Test x derivative
    expected = np.array(
        [
            [0.0, 0.01757465, 0.0, -0.01757465, 0.0],
            [0.0, 0.09211552, 0.0, -0.09211552, 0.0],
            [0.0, 0.10088742, 0.0, -0.10088742, 0.0],
            [0.0, 0.09211552, 0.0, -0.09211552, 0.0],
            [0.0, 0.01757465, 0.0, -0.01757465, 0.0],
        ]
    )
    for degree in (1, 2):
        assert_allclose(ndsavgol(z, (3, 3), degree, deriv=(0, 1)), expected, atol=1e-5)

    # Test y derivative
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01757465, 0.09211552, 0.10088742, 0.09211552, 0.01757465],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.01757465, -0.09211552, -0.10088742, -0.09211552, -0.01757465],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    for degree in (1, 2):
        assert_allclose(ndsavgol(z, (3, 3), degree, deriv=(1, 0)), expected, atol=1e-5)
