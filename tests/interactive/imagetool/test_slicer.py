import numpy as np
import xarray as xr
from qtpy import QtCore

from erlab.interactive.imagetool.slicer import ArraySlicer


def test_nonuniform_axes_ignores_user_idx_dim(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x_idx", "y"),
        coords={"x_idx": np.arange(3), "y": np.arange(4)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer._nonuniform_axes == []


def test_nonuniform_axes_detects_generated_idx_dim(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={"x": np.array([0.0, 1.0, 1.5]), "y": np.arange(4)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert str(slicer._obj.dims[0]).endswith("_idx")
    assert slicer._nonuniform_axes == [0]
