import lmfit
import numpy as np
import scipy.ndimage
from numpy.testing import assert_approx_equal

from erlab.analysis.fit.functions import lorentzian, lorentzian_wh
from erlab.analysis.fit.minuit import Minuit
from erlab.analysis.fit.models import MultiPeakModel


def test_minuit_from_lmfit() -> None:
    # Generate 2 lorentzian peaks with poisson noise
    xval = np.linspace(-1, 1, 250)

    yval = lorentzian_wh(xval, -0.5, 0.1, 10)
    yval += lorentzian_wh(xval, 0.5, 0.1, 10)

    # Convolve with gaussian
    yval = scipy.ndimage.gaussian_filter1d(yval, 5)

    npts = 10000
    rng = np.random.default_rng(1)
    yerr = 1 / np.sqrt(npts)
    yval = rng.poisson(yval * npts).astype(float)

    # lmfit model
    model = MultiPeakModel(
        npeaks=2, peak_shapes=["lorentzian"], background="none", fd=False, convolve=True
    )

    m = Minuit.from_lmfit(model, yval, xval, yerr, p0_center=-0.5, p1_center=0.5)
    m.scipy()

    assert_approx_equal(np.abs(m.values["p0_center"]), 0.5, 2)
    assert_approx_equal(np.abs(m.values["p1_center"]), 0.5, 2)

    assert m._repr_html_().startswith("<table>\n")


def test_minuit_from_lmfit_composite() -> None:
    # Generate 2 lorentzian peaks with poisson noise
    xval = np.linspace(-1, 1, 250)

    yval = 2 * xval + 4
    yval += lorentzian(xval, -0.5, 0.05, 10)
    yval += lorentzian(xval, 0.5, 0.05, 10)
    yval /= yval.sum()

    # Add noise
    npts = 100000
    rng = np.random.default_rng(1)
    yerr = 1 / np.sqrt(npts)
    yval = rng.poisson(yval * npts).astype(float)

    # lmfit model
    model = (
        lmfit.models.LorentzianModel(prefix="p0_")
        + lmfit.models.LorentzianModel(prefix="p1_")
        + lmfit.models.LinearModel()
    )

    m = Minuit.from_lmfit(
        model,
        yval,
        xval,
        yerr,
        p0_amplitude=10,
        p1_amplitude=10,
        p0_sigma=0.05,
        p1_sigma=0.05,
        p0_center=-0.5,
        p1_center=0.5,
        slope={"value": 2, "min": 0},
        intercept={"value": 4, "min": 0},
    )
    m.scipy()
    m.simplex()
    m.migrad()
    m.minos()
    m.hesse()

    assert_approx_equal(np.abs(m.values["p0_center"]), 0.5, 2)
    assert_approx_equal(np.abs(m.values["p1_center"]), 0.5, 2)

    assert m._repr_html_().startswith("<table>\n")
