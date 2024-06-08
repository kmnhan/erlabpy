import numpy as np
import scipy.ndimage
from erlab.analysis.fit.functions import lorentzian_wh
from erlab.analysis.fit.minuit import Minuit
from erlab.analysis.fit.models import MultiPeakModel
from numpy.testing import assert_approx_equal


def test_minuit_from_lmfit():
    # Generate 2 lorentzian peaks with poisson noise
    xval = np.linspace(-1, 1, 250)

    yval = lorentzian_wh(xval, -0.5, 0.1, 10)
    yval += lorentzian_wh(xval, 0.5, 0.1, 10)
    yval /= yval.sum()

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
    m.migrad()
    m.minos()
    m.hesse()

    assert_approx_equal(np.abs(m.values["p0_center"]), 0.5, 2)
    assert_approx_equal(np.abs(m.values["p1_center"]), 0.5, 2)

    assert m._repr_html_().startswith("<table>\n")
