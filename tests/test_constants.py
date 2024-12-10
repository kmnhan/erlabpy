from numpy.testing import assert_almost_equal

from erlab.constants import conv_eV_nm, conv_watt_photons


def test_conv_watt_photons() -> None:
    assert_almost_equal(conv_watt_photons(1e-9, 20), 100682331.35085419)


def test_conv_ev_nm() -> None:
    assert_almost_equal(conv_eV_nm(1.0), 1239.8419843320028)
