import numpy as np
import xarray as xr

from erlab.interactive.curvefittingtool import edctool, mdctool


def test_edc_mdctool(qtbot) -> None:
    data = xr.DataArray(np.arange(25), dims=["x"]).astype(np.float64)
    win_mdc = mdctool(data, execute=False)
    win_edc = edctool(data, execute=False)
    qtbot.addWidget(win_edc)
    qtbot.addWidget(win_mdc)
