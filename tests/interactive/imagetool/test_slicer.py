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


def test_set_array_shallow_copy_does_not_require_deep_copy(qtbot) -> None:
    data1 = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    data2 = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
    )

    slicer = ArraySlicer(data1, parent=QtCore.QObject())
    slicer.set_array(data2, validate=False, reset=True)

    assert slicer._obj.equals(data2)


def test_validate_array_does_not_deepcopy_attrs(qtbot) -> None:
    class _NoDeepCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("deepcopy should not be called")

    sentinel = _NoDeepCopy()
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
        attrs={"sentinel": sentinel},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    assert slicer._obj.attrs["sentinel"] is sentinel


def test_index_of_value_nonuniform_descending_axis(qtbot) -> None:
    data = xr.DataArray(
        np.zeros((4, 3)),
        dims=("x", "y"),
        coords={"x": np.array([5.0, 3.0, 2.0, -1.0]), "y": np.arange(3)},
    )

    slicer = ArraySlicer(data, parent=QtCore.QObject())

    for value in (6.0, 4.2, 2.4, 0.0, -2.0):
        idx = slicer.index_of_value(0, value, uniform=False)
        expected = int(np.argmin(np.abs(data.x.values - value)))
        assert idx == expected
