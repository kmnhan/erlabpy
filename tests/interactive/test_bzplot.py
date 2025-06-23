import numpy as np
from numpy.testing import assert_allclose

from erlab.interactive.bzplot import BZPlotter


def test_bzplot(qtbot) -> None:
    win = BZPlotter(execute=False)
    qtbot.addWidget(win)
    qtbot.addWidget(win.controls)

    win.controls.params_latt.set_values(
        a=3.0, b=3.0, c=6.0, alpha=90.0, beta=90.0, gamma=120.0
    )
    win.controls.params_latt.widgets["apply"].click()

    assert_allclose(win.controls.params_avec.values["a1x"], 3.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a1y"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a1z"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a2x"], -1.5, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a2z"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a3x"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a3y"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a3z"], 6.0, atol=1e-15)
    assert_allclose(
        win.controls.params_bvec.values["b1y"], 2 * np.pi / np.sqrt(3) / 3, atol=1e-15
    )
    assert_allclose(win.controls.params_bvec.values["b1z"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_bvec.values["b2x"], 0.0, atol=1e-15)
    assert_allclose(
        win.controls.params_bvec.values["b2y"], 4 * np.pi / np.sqrt(3) / 3, atol=1e-15
    )
    assert_allclose(win.controls.params_bvec.values["b2z"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_bvec.values["b3x"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_bvec.values["b3x"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_bvec.values["b3y"], 0.0, atol=1e-15)

    win.controls.close()
    win.close()
