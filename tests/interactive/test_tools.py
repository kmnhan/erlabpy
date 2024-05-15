import numpy as np
import pyperclip
import xarray as xr
from erlab.interactive.derivative import dtool
from erlab.interactive.fermiedge import goldtool

from ..analysis.test_gold import gold  # noqa: TID252, F401


def test_goldtool(qtbot, gold):  # noqa: F811
    win = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    with qtbot.waitActive(win):
        win.show()
        win.activateWindow()
        win.raise_()


def test_dtool(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]).astype(
        np.float64
    )
    win = dtool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitActive(win):
        win.show()
        win.activateWindow()
        win.raise_()

    win.tab_widget.setCurrentIndex(0)
    win.interp_group.setChecked(False)
    win.smooth_group.setChecked(True)
    win.copy_btn.click()
    assert (
        pyperclip.paste()
        == """_processed = data.copy()
for _ in range(1):
\t_processed = era.image.gaussian_filter(_processed, sigma={\"y\": 1.0, \"x\": 1.0})
result = _processed.differentiate('y').differentiate('y')"""
    )

    win.smooth_group.setChecked(False)

    win.copy_btn.click()
    assert pyperclip.paste() == "result = data.differentiate('y').differentiate('y')"

    win.tab_widget.setCurrentIndex(1)
    win.copy_btn.click()
    assert pyperclip.paste() == "result = era.image.scaled_laplace(data, factor=1.0)"

    win.tab_widget.setCurrentIndex(2)
    win.copy_btn.click()
    assert pyperclip.paste() == "result = era.image.curvature(data, a0=1.0, factor=1.0)"

    win.tab_widget.setCurrentIndex(3)
    win.copy_btn.click()
    assert pyperclip.paste() == "result = era.image.minimum_gradient(data)"
